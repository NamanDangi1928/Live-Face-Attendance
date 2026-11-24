"""
Run ONCE from the project folder (the one with app.py):

    python apply_camera_patch.py

Adds to templates/dashboard_new.html:
  * a camera picker (built-in webcam / DroidCam / any camera) on BOTH cards
  * a Mobile / Laptop switch on the Classroom card (it had none)
  * a classroom video element (it was missing, so laptop mode could not work)
  * frames keep their real shape (no stretching) and the first classroom
    snapshot is sent after ~2 s instead of 10 s
Makes a backup first. Safe to run twice.
"""
import re, shutil, sys
from pathlib import Path

path = Path("templates/dashboard_new.html")
if not path.exists():
    sys.exit("templates/dashboard_new.html not found. Run this from the folder that contains app.py")

src = path.read_text(encoding="utf-8")
bak = Path("templates/dashboard_new.html.bak2")
if not bak.exists():
    shutil.copy(path, bak)
    print("Backup saved as templates/dashboard_new.html.bak2")

report = []
def done(name, ok): report.append((name, ok))

SEL_STYLE = ("max-width:150px;font-size:0.72rem;padding:4px 6px;border-radius:8px;"
             "border:1px solid var(--border-subtle,#e2e8f0);background:var(--bg-tertiary,#f1f5f9);"
             "color:var(--text-primary,#1e293b);")

GATE_SELECT = (f'<select id="gate-device" title="Choose camera" onchange="onDeviceChange(\'gate\')" '
               f'style="{SEL_STYLE}"></select>\n                    ')

CLASS_CONTROLS = (
    '<div class="camera-source-toggle">\n'
    '                        <button class="source-btn active" id="classroom-mobile-btn" '
    'onclick="switchSource(\'classroom\',\'mobile\')">Mobile</button>\n'
    '                        <button class="source-btn" id="classroom-laptop-btn" '
    'onclick="switchSource(\'classroom\',\'laptop\')">Laptop</button>\n'
    '                    </div>\n'
    f'                    <select id="classroom-device" title="Choose camera" '
    f'onchange="onDeviceChange(\'classroom\')" style="{SEL_STYLE}"></select>\n'
    '                    '
)

CLASS_VIDEO = ('<video id="classroom-local-video" class="local-video" autoplay playsinline muted></video>\n'
               '                    ')

OVERRIDES = r'''
    <script>
        // ===== Camera picker + classroom camera (overrides the older functions above) =====

        async function loadCameras() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
            try {
                // Ask permission once, otherwise the browser hides the camera names
                try {
                    const t = await navigator.mediaDevices.getUserMedia({ video: true });
                    t.getTracks().forEach(x => x.stop());
                } catch (e) { console.warn('Camera permission:', e); }

                const devices = (await navigator.mediaDevices.enumerateDevices())
                    .filter(d => d.kind === 'videoinput');
                ['gate', 'classroom'].forEach(room => {
                    const sel = document.getElementById(`${room}-device`);
                    if (!sel) return;
                    const prev = sel.value;
                    sel.innerHTML = '';
                    devices.forEach((d, i) => {
                        const o = document.createElement('option');
                        o.value = d.deviceId;
                        o.textContent = d.label || `Camera ${i + 1}`;
                        sel.appendChild(o);
                    });
                    if (prev) sel.value = prev;
                    else if (room === 'classroom' && devices.length > 1) sel.selectedIndex = 1;
                });
            } catch (e) { console.error('Could not list cameras:', e); }
        }
        document.addEventListener('DOMContentLoaded', loadCameras);

        async function onDeviceChange(room) {
            const state = localState[room];
            if (state.mode === 'laptop') {
                stopLocalCamera(room);
                state.mode = 'mobile';          // force a fresh start with the new camera
                await switchSource(room, 'laptop');
            }
        }

        async function switchSource(room, mode) {
            const state = localState[room];
            if (state.mode === mode && (mode !== 'laptop' || state.stream)) return;
            state.mode = mode;

            const setToggle = (m) => {
                const mb = document.getElementById(`${room}-mobile-btn`);
                const lb = document.getElementById(`${room}-laptop-btn`);
                if (mb) mb.classList.toggle('active', m === 'mobile');
                if (lb) lb.classList.toggle('active', m === 'laptop');
            };
            setToggle(mode);

            const localVideo = document.getElementById(`${room}-local-video`);
            const snapshot = document.getElementById(`${room}-snapshot`);
            const placeholder = document.getElementById(`${room}-placeholder`);
            const label = document.getElementById(`${room}-mode-label`);
            const niceName = room === 'gate' ? 'Gate' : 'Classroom';

            if (mode === 'laptop') {
                stopLocalCamera(room);
                const sel = document.getElementById(`${room}-device`);
                const video = { width: { ideal: 640 }, height: { ideal: 480 } };
                if (sel && sel.value) video.deviceId = { exact: sel.value };
                else video.facingMode = 'user';
                try {
                    state.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
                    localVideo.srcObject = state.stream;
                    localVideo.style.display = 'block';
                    await localVideo.play();
                    if (snapshot) snapshot.style.display = 'none';
                    if (placeholder) placeholder.style.display = 'none';
                    updateStatus(room, true);
                    startLocalFrameSending(room, localVideo);
                    const camName = sel && sel.selectedOptions[0] ? sel.selectedOptions[0].textContent : 'Laptop camera';
                    if (label) label.textContent = camName;
                    showToast(`${niceName}: using ${camName}`, 'success');
                } catch (err) {
                    showToast(`Could not open camera (${err.name}). Pick another camera or close other apps using it.`, 'error');
                    state.mode = 'mobile';
                    setToggle('mobile');
                    stopLocalCamera(room);
                    if (localVideo) localVideo.style.display = 'none';
                    if (placeholder) placeholder.style.display = 'flex';
                }
            } else {
                stopLocalCamera(room);
                if (localVideo) localVideo.style.display = 'none';
                if (placeholder) placeholder.style.display = 'flex';
                if (label) label.textContent = room === 'gate' ? 'WebRTC Live Stream' : 'High-Quality Snapshots';
                showToast(`${niceName}: switched to mobile camera`, 'success');
            }
        }

        function startLocalFrameSending(room, videoEl) {
            const state = localState[room];
            if (state.frameInterval) clearInterval(state.frameInterval);

            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const isClassroom = (room === 'classroom');
            const interval = isClassroom ? 10000 : 200;           // classroom: every 10 s, gate: 5 fps
            const msgType = isClassroom ? 'classroom_snapshot' : 'process_frame';

            const sendFrame = () => {
                if (!ws || ws.readyState !== WebSocket.OPEN) return;
                if (!videoEl.videoWidth) return;                    // camera not ready yet
                if (videoEl.paused) videoEl.play().catch(() => { });
                try {
                    // keep the camera's real shape (no stretching)
                    canvas.width = 640;
                    canvas.height = Math.round(640 * videoEl.videoHeight / videoEl.videoWidth);
                    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
                    const dataUrl = canvas.toDataURL('image/jpeg', isClassroom ? 0.9 : 0.4);
                    ws.send(JSON.stringify({
                        type: msgType,
                        frame: dataUrl,
                        timestamp: new Date().toISOString(),
                        source: 'laptop_dashboard'
                    }));
                } catch (e) { console.error('Frame send error:', e); }
            };

            state.frameInterval = setInterval(sendFrame, interval);
            if (isClassroom) setTimeout(sendFrame, 2000);           // first snapshot after ~2 s
        }

        function handleClassroomSnapshot(data) {
            const img = document.getElementById('classroom-snapshot');
            const placeholder = document.getElementById('classroom-placeholder');
            const localVideo = document.getElementById('classroom-local-video');
            if (img && data.image) {
                img.src = `data:image/jpeg;base64,${data.image}`;
                img.style.cssText = 'display:block;width:100%;height:100%;object-fit:cover;';
                if (placeholder) placeholder.style.display = 'none';
                if (localVideo) localVideo.style.display = 'none';   // show the annotated snapshot instead
            }
            document.getElementById('classroom-count').textContent = `${data.student_count || 0} Students`;
            document.getElementById('classroom-timestamp').textContent = new Date().toLocaleTimeString();
            updateStatus('classroom', true);
        }
    </script>
'''

# 1. camera picker on the gate card
if 'id="gate-device"' not in src:
    src, n = re.subn(r'<span id="gate-face-count">', lambda m: GATE_SELECT + m.group(0), src, count=1)
    done("1 gate camera picker", n == 1)
else:
    done("1 gate picker (already done)", True)

# 2. classroom toggle + picker
if 'id="classroom-device"' not in src:
    src, n = re.subn(r'<span id="classroom-count">', lambda m: CLASS_CONTROLS + m.group(0), src, count=1)
    done("2 classroom Mobile/Laptop switch + picker", n == 1)
else:
    done("2 classroom controls (already done)", True)

# 3. classroom video element
if 'id="classroom-local-video"' not in src:
    src, n = re.subn(r'<canvas id="classroom-overlay"></canvas>', lambda m: CLASS_VIDEO + m.group(0), src, count=1)
    done("3 classroom video element", n == 1)
else:
    done("3 classroom video (already done)", True)

# 4. override script just before </body>
if "function loadCameras" not in src:
    src, n = re.subn(r'</body>', lambda m: OVERRIDES + '</body>', src, count=1)
    done("4 camera picker / classroom logic", n == 1)
else:
    done("4 logic (already done)", True)

path.write_text(src, encoding="utf-8")
print()
for name, ok in report:
    print(("OK      " if ok else "FAILED  ") + name)
print("\nAll edits applied." if all(ok for _, ok in report) else "\nSome edits FAILED - send me this output.")
print("Now refresh the dashboard with Ctrl+Shift+R.")