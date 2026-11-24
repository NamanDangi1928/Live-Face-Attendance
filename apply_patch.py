"""
Run ONCE from the project folder (the folder that contains app.py):
 
    python apply_patch.py
 
It makes a backup (app.py.bak) and then edits app.py. Safe to run twice:
it skips edits that are already done.
"""
import re, shutil, sys
from pathlib import Path
 
path = Path("app.py")
if not path.exists():
    sys.exit("app.py not found. Run this from the folder that contains app.py")
 
src = path.read_text(encoding="utf-8")
if not Path("app.py.bak").exists():
    shutil.copy("app.py", "app.py.bak")
    print("Backup saved as app.py.bak")
 
report = []
 
def done(name, ok):
    report.append((name, ok))
 
 
# ---- 1. state: add a per-student match counter ---------------------------
if "self.streaks" not in src:
    new, n = re.subn(
        r"^([ \t]*)(self\.last_announced = \{\}.*)$",
        r"\1\2\n\1self.streaks = {}  # {student_id: {'n': matches in a row, 't': last time}}",
        src, count=1, flags=re.M)
    src = new; done("1 add streak counter to state", n == 1)
else:
    done("1 add streak counter to state (already done)", True)
 
if "MATCH_STREAK_NEEDED" not in src:
    new, n = re.subn(
        r"^_state = AttendanceState\(\)\s*$",
        "_state = AttendanceState()\n\n"
        "# A student is only marked after this many matches in a row (stops name flipping)\n"
        "MATCH_STREAK_NEEDED = 3",
        src, count=1, flags=re.M)
    src = new; done("2 add MATCH_STREAK_NEEDED", n == 1)
else:
    done("2 add MATCH_STREAK_NEEDED (already done)", True)
 
 
# ---- 2. gate camera: aligned recognition + streak ------------------------
if "recognize_async(frame, detection)" not in src:
    pattern = (r"^([ \t]*)face_img = face_detector\.extract_face\(frame, detection\)\s*"
               r"if face_img is None:\s*names_list\.append\(\"Unknown\"\)\s*"
               r"statuses_list\.append\(\"unknown\"\)\s*continue\s*"
               r"# Recognition\s*"
               r"match = await face_recognizer\.recognize_async\(face_img\)")
 
    def _gate_repl(m):
        I = m.group(1)
        return (
            f"{I}# Recognition (face is aligned using YuNet landmarks)\n"
            f"{I}match = await face_recognizer.recognize_async(frame, detection)\n"
            f"{I}pending = False\n"
            f"{I}if match:\n"
            f"{I}    streak = _state.streaks.get(match.student_id, {{'n': 0, 't': 0.0}})\n"
            f"{I}    if current_time - streak['t'] > 3:\n"
            f"{I}        streak['n'] = 0\n"
            f"{I}    streak['n'] += 1\n"
            f"{I}    streak['t'] = current_time\n"
            f"{I}    _state.streaks[match.student_id] = streak\n"
            f"{I}    if streak['n'] < MATCH_STREAK_NEEDED:\n"
            f"{I}        pending = True   # not sure yet, wait for more matches\n"
            f"{I}        match = None"
        )
 
    src, n = re.subn(pattern, _gate_repl, src, count=1, flags=re.M)
    done("3 gate: aligned recognition + 3-in-a-row", n == 1)
else:
    done("3 gate: aligned recognition (already done)", True)
 
if "Verifying..." not in src:
    src, n = re.subn(r'name = "Unknown"(\s*)confidence = 0\.0(\s*)status = "unknown"',
                     r'name = "Verifying..." if pending else "Unknown"\1confidence = 0.0\2status = "unknown"',
                     src, count=1)
    done("4 gate: show 'Verifying...' label", n == 1)
else:
    done("4 gate: label (already done)", True)
 
if "not pending and current_time" not in src:
    src, n = re.subn(r"if current_time - _state\.last_unknown_alert > 30",
                     "if not pending and current_time - _state.last_unknown_alert > 30",
                     src, count=1)
    done("5 gate: no alarm while still verifying", n == 1)
else:
    done("5 gate: alarm rule (already done)", True)
 
 
# ---- 3. classroom snapshot -----------------------------------------------
if "face_recognizer.recognize(frame, det)" not in src:
    src, n = re.subn(r"face_recognizer\.recognize\(face_crop\)",
                     "face_recognizer.recognize(frame, det)", src, count=1)
    done("6 classroom: aligned recognition", n == 1)
else:
    done("6 classroom (already done)", True)
 
 
# ---- 4. 3-step enrol endpoint --------------------------------------------
if "face_detector.detect(image, apply_smoothing=False)" not in src:
    src, n = re.subn(
        r"^([ \t]*)embedding = face_recognizer\.get_embedding\(image\)\s*$",
        lambda m: (
            f"{m.group(1)}_dets = face_detector.detect(image, apply_smoothing=False)\n"
            f"{m.group(1)}embedding = None\n"
            f"{m.group(1)}if _dets:\n"
            f"{m.group(1)}    _det = max(_dets, key=lambda d: d.width * d.height)\n"
            f"{m.group(1)}    embedding = face_recognizer.embed(image, _det)"),
        src, count=1, flags=re.M)
    done("7 3-step enrol: aligned embedding", n == 1)
else:
    done("7 3-step enrol (already done)", True)
 
if "_load_enrolled_faces()" not in src:
    src, n = re.subn(
        r"^([ \t]*)(success = await mongodb\.add_face_embedding\(student_id, embedding, image_path\))\s*$",
        lambda m: (f"{m.group(1)}{m.group(2)}\n"
                   f"{m.group(1)}if success:\n"
                   f"{m.group(1)}    await face_recognizer._load_enrolled_faces()"),
        src, count=1, flags=re.M)
    done("8 3-step enrol: reload faces so no restart needed", n == 1)
else:
    done("8 reload (already done)", True)
 
 
# ---- 5. replace /api/enroll-with-files -----------------------------------
NEW_ENROLL = '''@app.post("/api/enroll-with-files")
async def enroll_with_files(
    name: str = Form(...),
    roll_number: str = Form(...),
    email: str = Form(None),
    phone: str = Form(None),
    images: List[UploadFile] = File(...)
):
    """Enroll a student: ONE aligned embedding PER photo (no averaging)."""
    try:
        collected = []  # list of (embedding, face_crop)
 
        # 1. Check the photos BEFORE creating the student
        for idx, file in enumerate(images):
            if idx >= 5:
                break
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue
 
            detections = face_detector.detect(img, apply_smoothing=False)
            if not detections:
                print(f"No face detected in image {idx}")
                continue
 
            det = max(detections, key=lambda d: d.width * d.height)
            embedding = face_recognizer.embed(img, det)   # aligned
            if embedding is None:
                continue
            collected.append((embedding, face_detector.extract_face(img, det)))
 
        if not collected:
            return JSONResponse(status_code=400, content={
                "success": False, "message": "No valid faces found in uploaded images."})
 
        # 2. Create the student and store every photo's embedding
        student = await mongodb.create_student(name, roll_number, email, phone)
        student_id = str(student["_id"])
 
        embeddings = []
        for i, (emb, crop) in enumerate(collected):
            crop_path = None
            if crop is not None:
                crop_path = str(config.STUDENTS_DIR / f"{student_id}_face{i + 1}.jpg")
                cv2.imwrite(crop_path, crop)
            await mongodb.add_face_embedding(student_id, emb.reshape(1, 128), crop_path)
            embeddings.append(emb)
 
        # 3. Update live recognizer immediately
        face_recognizer.set_student_embeddings(student_id, name, roll_number, embeddings)
 
        return {
            "success": True,
            "message": f"Student enrolled with {len(embeddings)} face photos."
        }
 
    except Exception as e:
        logger.error(f"Error enrolling with files: {e}")
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})
 
 
'''
start = src.find('@app.post("/api/enroll-with-files")')
end = src.find("# ----- Legacy Endpoints Removed -----")
if start != -1 and end != -1 and start < end:
    src = src[:start] + NEW_ENROLL + src[end:]
    done("9 enrol-with-files rewritten", True)
else:
    done("9 enrol-with-files rewritten", False)
 
path.write_text(src, encoding="utf-8")
print()
for name, ok in report:
    print(("OK      " if ok else "FAILED  ") + name)
if all(ok for _, ok in report):
    print("\nAll edits applied.")
else:
    print("\nSome edits FAILED - send me this output.")
 