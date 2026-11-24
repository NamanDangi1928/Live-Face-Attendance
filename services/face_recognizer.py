"""
Face Recognition Service using OpenCV SFace (DNN Embeddings)
 
FIXED VERSION:
  1. Faces are now ALIGNED (using the 5 YuNet landmarks) before SFace sees them.
     SFace was trained on aligned faces - skipping this was the main cause of
     "Unknown" and names flipping between students.
  2. Every enrolled photo is kept as its own embedding (no averaging of
     front/left/right), and each student is scored by their BEST photo.
  3. A match must beat the runner-up student by a margin, so two similar
     scores no longer flip back and forth.
  4. Added remove_student() (app.py called it but it did not exist).
"""
import asyncio
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2
 
import config
 
 
@dataclass
class RecognitionMatch:
    """Result of a face recognition match"""
    student_id: Union[int, str]
    name: str
    roll_number: str
    similarity: float
    margin: float = 1.0  # best score minus 2nd-best student's score
 
 
class FaceRecognizer:
    def __init__(self):
        self._recognizer = None
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._enrolled_faces = []
        self._embedding_matrix: Optional[np.ndarray] = None
        self._student_info: List[dict] = []
 
        # OpenCV recommends ~0.363 for SFace. config.py can override.
        self._confidence_threshold = getattr(config, 'FACE_RECOGNITION_THRESHOLD', 0.38)
        # Winner must beat the 2nd-best student by at least this much.
        self._min_margin = getattr(config, 'FACE_MATCH_MARGIN', 0.05)
 
    # ------------------------------------------------------------------ setup
    def initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            model_path = str(config.SFACE_MODEL_PATH)
            if not config.SFACE_MODEL_PATH.exists():
                print(f"❌ SFace model not found at {model_path}")
                return False
 
            if hasattr(cv2, 'FaceRecognizerSF_create'):
                create_fn = cv2.FaceRecognizerSF_create
            elif hasattr(cv2, 'face') and hasattr(cv2.face, 'FaceRecognizerSF_create'):
                create_fn = cv2.face.FaceRecognizerSF_create
            else:
                print("❌ FaceRecognizerSF_create not found in cv2 or cv2.face")
                return False
 
            self._recognizer = create_fn(model_path, "")
            self._initialized = True
            print("✅ Face Recognizer initialized (SFace DNN, aligned)")
            print(f"   Confidence threshold: {self._confidence_threshold}")
            print(f"   Minimum margin over 2nd student: {self._min_margin}")
 
            try:
                asyncio.get_running_loop()
                asyncio.create_task(self._load_enrolled_faces())
            except RuntimeError:
                print("   Embeddings will be loaded when app starts")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Face Recognizer: {e}")
            return False
 
    async def _load_enrolled_faces(self):
        """Load every stored embedding from MongoDB (one row per photo)."""
        try:
            from database import mongodb
            print("[FACE] Loading embeddings from MongoDB...")
            all_students = await mongodb.get_all_students()
 
            faces = []
            count = 0
            for student in all_students:
                student_id = str(student.get("_id"))
                name = student.get("name", "Unknown")
                roll_number = student.get("roll_number", "")
                for emb_list in student.get("face_embeddings", []):
                    embedding = np.array(emb_list, dtype=np.float32).flatten()
                    if embedding.size != 128:
                        continue  # skip empty/dummy entries
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    faces.append({
                        "embedding": embedding,
                        "student_id": student_id,
                        "name": name,
                        "roll_number": roll_number,
                        "filename": "db_record"
                    })
                    count += 1
 
            self._enrolled_faces = faces
            self._rebuild_embedding_matrix()
            print(f"[FACE] Total enrolled: {count} embeddings from {len(all_students)} students")
        except Exception as e:
            print(f"[FACE] Error loading faces: {e}")
            import traceback
            traceback.print_exc()
 
    def _rebuild_embedding_matrix(self):
        if not self._enrolled_faces:
            self._embedding_matrix = None
            self._student_info = []
            return
        rows = []
        self._student_info = []
        for face in self._enrolled_faces:
            emb = np.asarray(face["embedding"], dtype=np.float32).flatten()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            rows.append(emb)
            self._student_info.append({
                "student_id": face["student_id"],
                "name": face["name"],
                "roll_number": face["roll_number"]
            })
        self._embedding_matrix = np.vstack(rows)
 
    # ------------------------------------------------------------- embeddings
    def embed(self, frame: np.ndarray, detection) -> Optional[np.ndarray]:
        """
        Aligned embedding. `frame` is the FULL image, `detection` is the
        FaceDetection from face_detector (needs .bbox and .landmarks).
        """
        if self._recognizer is None or detection is None:
            return None
        if getattr(detection, "landmarks", None) is None:
            return None
        try:
            x, y, w, h = detection.bbox
            # Same layout YuNet outputs: x,y,w,h, 5 landmark points, score
            row = np.concatenate([
                np.array([x, y, w, h], dtype=np.float32),
                np.asarray(detection.landmarks, dtype=np.float32).flatten(),
                np.array([detection.confidence], dtype=np.float32),
            ]).astype(np.float32)
 
            try:
                aligned = self._recognizer.alignCrop(frame, row)
            except Exception:
                aligned = self._recognizer.alignCrop(frame, row.reshape(1, -1))
 
            feature = self._recognizer.feature(aligned)
            if feature is None:
                return None
            feature = feature.flatten()
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            return feature
        except Exception as e:
            print(f"❌ Aligned embedding error: {type(e).__name__}: {e}")
            return None
 
    def _compute_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """OLD unaligned method. Only used as a fallback; less accurate."""
        if self._recognizer is None:
            return None
        try:
            if len(face_image.shape) == 2:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
            if face_image.dtype != np.uint8:
                face_image = face_image.astype(np.uint8)
            face_input = cv2.resize(face_image, (112, 112))
            feature = self._recognizer.feature(face_input)
            if feature is None:
                return None
            feature = feature.flatten()
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            return feature
        except Exception as e:
            print(f"❌ Embedding exception: {type(e).__name__}: {e}")
            return None
 
    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        return self._compute_embedding(face_image)
 
    # ------------------------------------------------------------ recognition
    def recognize(self, face_image: np.ndarray, detection=None) -> Optional[RecognitionMatch]:
        """
        Preferred call:  recognize(full_frame, detection)   -> aligned
        Fallback call:   recognize(face_crop)               -> unaligned
        """
        if not self._initialized:
            return None
        if self._embedding_matrix is None or len(self._student_info) == 0:
            return None
        try:
            if detection is not None:
                query = self.embed(face_image, detection)
            else:
                query = self._compute_embedding(face_image)
            if query is None:
                return None
 
            sims = np.dot(self._embedding_matrix, query)
 
            # Best score PER STUDENT (max over that student's photos)
            best_per_student = {}
            for i, s in enumerate(sims):
                sid = self._student_info[i]["student_id"]
                if sid not in best_per_student or s > best_per_student[sid][0]:
                    best_per_student[sid] = (float(s), i)
 
            ranked = sorted(best_per_student.values(), key=lambda t: t[0], reverse=True)
            best_score, best_idx = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else -1.0
            margin = best_score - second_score
            info = self._student_info[best_idx]
 
            print(f"[RECOGNIZE] best={info['name']} {best_score:.2f} | "
                  f"2nd={second_score:.2f} | margin={margin:.2f}")
 
            if best_score >= self._confidence_threshold and margin >= self._min_margin:
                return RecognitionMatch(
                    student_id=info["student_id"],
                    name=info["name"],
                    roll_number=info["roll_number"],
                    similarity=best_score,
                    margin=margin,
                )
            return None
        except Exception as e:
            print(f"Recognition error: {e}")
            import traceback
            traceback.print_exc()
            return None
 
    async def recognize_async(self, face_image: np.ndarray, detection=None) -> Optional[RecognitionMatch]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.recognize, face_image, detection)
 
    # ----------------------------------------------------- enrolment helpers
    def set_student_embeddings(self, student_id, name: str, roll_number: str,
                               embeddings: List[np.ndarray]):
        """Replace this student's embeddings in memory (one per photo)."""
        sid = str(student_id)
        self._enrolled_faces = [f for f in self._enrolled_faces if str(f["student_id"]) != sid]
        for emb in embeddings:
            self._enrolled_faces.append({
                "embedding": np.asarray(emb, dtype=np.float32).flatten(),
                "student_id": sid,
                "name": name,
                "roll_number": roll_number,
                "filename": "enrollment"
            })
        self._rebuild_embedding_matrix()
 
    def remove_student(self, student_id):
        sid = str(student_id)
        self._enrolled_faces = [f for f in self._enrolled_faces if str(f["student_id"]) != sid]
        self._rebuild_embedding_matrix()
 
    def enroll_student(self, student_id, name, roll_number, face_image) -> Tuple[bool, str]:
        """Legacy (unaligned) single-image enrol. Prefer set_student_embeddings."""
        feature = self._compute_embedding(face_image)
        if feature is None:
            return False, "Could not extract features"
        self._enrolled_faces.append({
            "embedding": feature, "student_id": str(student_id), "name": name,
            "roll_number": roll_number, "filename": "memory_enrollment"
        })
        self._rebuild_embedding_matrix()
        return True, "Enrolled in memory"
 
    def get_enrolled_count(self) -> int:
        return len(self._enrolled_faces)
 
    def close(self):
        self._executor.shutdown(wait=False)
 
 
face_recognizer = FaceRecognizer()
 

