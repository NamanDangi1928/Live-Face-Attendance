# EYE-X: Complete Technical Documentation
## Smart Attendance System - Comprehensive File-by-File Guide

---

# System Overview

**EYE-X** is an AI-powered real-time attendance system using:
- **YuNet DNN** for face detection
- **SFace DNN** for face recognition (128D embeddings)
- **YOLOv8-pose** for behavior detection
- **WebSocket** for real-time communication
- **FastAPI** async web framework
- **MongoDB Atlas** for data storage

---

# 1. ROOT CONFIGURATION FILES

## 1.1 `config.py` - Application Configuration

**Purpose**: Centralizes all configuration settings for the application.

**Key Settings**:
```python
# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STUDENTS_DIR = DATA_DIR / "students"

# Server
HOST = "0.0.0.0"
PORT = 8000

# Database
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = "eye_x"

# AI Models
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
YOLO_MODEL = "yolov8n-pose.pt"

# Thresholds
YUNET_SCORE_THRESHOLD = 0.7      # Face detection confidence
FACE_RECOGNITION_THRESHOLD = 0.40  # Cosine similarity for recognition
```

**How it works**:
1. Loads environment variables from `.env` file
2. Creates necessary directories on import
3. Provides paths to AI models
4. Sets detection/recognition thresholds

---

## 1.2 `requirements.txt` - Python Dependencies

**Core Dependencies**:
- `fastapi>=0.104.0` - Modern async web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `opencv-contrib-python>=4.8.0` - Computer vision (YuNet, SFace)
- `ultralytics>=8.0.0` - YOLOv8 for pose estimation
- `motor>=3.3.0` - Async MongoDB driver
- `websockets>=12.0` - WebSocket protocol
- `numpy>=1.24.0` - Numerical operations
- `pyttsx3>=2.90` - Text-to-speech (client-side)

**Downloaded Modules Explained**:
When you run `pip install -r requirements.txt`, these modules are installed:

1. **FastAPI** → Web framework for REST APIs and WebSocket endpoints
2. **OpenCV** → Face detection (YuNet) and recognition (SFace)
3. **Ultralytics** → YOLOv8 pose estimation for behavior analysis
4. **Motor** → Async MongoDB operations for storing attendance
5. **NumPy** → Matrix operations for embedding comparisons
6. **Uvicorn** → Production-ready server to run the app

---

# 2. MAIN APPLICATION FILE

## 2.1 `app.py` - FastAPI Server (1265 lines)

**Purpose**: Main application entry point handling all routes, WebSocket connections, and AI processing.

### Structure Overview:

#### **Lifespan Management (Lines 44-75)**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await mongodb.connect_mongodb()
    face_detector.initialize()
    face_recognizer.initialize()
    behavior_detector.initialize()
    audio_service.initialize()
    
    yield
    
    # Shutdown
    await mongodb.close_mongodb()
    face_detector.close()
```

**How it works**: Executes on server startup to initialize all AI models and database connections.

---

#### **Page Routes (Lines 103-149)**

| Route | Template | Purpose |
|-------|----------|---------|
| `/` | `landing.html` | Marketing homepage |
| `/dashboard` | `dashboard_new.html` | Admin console with live feeds |
| `/add-students` | `add_students.html` | Enroll new students |
| `/analysis` | `analysis.html` | Charts and analytics |
| `/data` | `data.html` | Student records and history |
| `/gate` | `gate_camera.html` | Mobile gate camera |
| `/classroom` | `classroom_camera.html` | Mobile classroom camera |
| `/settings` | `settings.html` | Configure lecture schedules |

---

#### **WebSocket Endpoint (Lines 158-274)**

**Route**: `/ws/{room}` where room = "gate", "classroom", or "dashboard"

**How it works**:
1. Client connects to WebSocket
2. Server assigns client to a room
3. Handles signaling messages:
   - **P2P Signaling**: `offer`, `answer`, `ice-candidate` for WebRTC
   - **Frame Processing**: `process_frame`, `classroom_snapshot`
   - **Ping/Pong**: Keep-alive messages

**Message Flow**:
```
Gate Camera → process_frame → Server processes → Broadcasts to dashboard
Classroom → classroom_snapshot → Behavior detection → Broadcasts results
Dashboard ← Receives snapshots with annotations ← Server
```

---

#### **Core Processing Functions**

### `process_frame()` (Lines 389-542)
**Purpose**: Process gate camera frames for face recognition and attendance.

**Algorithm**:
```
1. Decode base64 JPEG frame
2. Detect faces using YuNet
3. For each detected face:
   a. Extract face region
   b. Run SFace recognition
   c. If match found:
      - Check if already marked today (duplicate prevention)
      - Mark attendance in MongoDB
      - Broadcast notification to dashboard
   d. If unknown:
      - Trigger intrusion alert (debounced every 30s)
4. Draw bounding boxes on frame
5. Send annotated frame to dashboard
```

---

### `process_classroom_snapshot()` (Lines 593-723)
**Purpose**: Process classroom camera for both face recognition and behavior detection.

**Algorithm**:
```
1. Decode frame
2. Run face detection + recognition (for attendance)
3. Run YOLOv8-pose detection (for behavior)
4. Draw annotations:
   - Green boxes for recognized faces
   - Colored boxes for behaviors:
     * Green: Studying
     * Blue: Focused
     * Orange: Distracted
     * Red: Sleeping
5. Encode annotated frame
6. Broadcast to dashboard
7. Log behavior stats to MongoDB (rate-limited to every 10s)
```

---

#### **REST API Endpoints**

### Student Management
```python
GET  /api/students              # List all students
POST /api/mongodb/students      # Create new student
POST /api/mongodb/students/{id}/enroll-face  # Add face embedding
POST /api/enroll-with-files     # Enroll with multiple images (averages embeddings)
DELETE /api/students/{id}       # Deactivate student
```

### Attendance
```python
GET  /api/attendance/today      # Today's attendance records
GET  /api/attendance/stats      # Total, present, absent counts
POST /api/attendance/manual     # Manually mark attendance
```

### Analytics
```python
GET /api/analytics/attendance-trend    # 7-day trend
GET /api/analytics/behavior-summary    # Behavior distribution
GET /api/analytics/hourly-behavior     # Hourly breakdown
GET /api/analytics/leaderboard         # Top students by attendance
```

### Export
```python
GET /api/export/daily?date=2026-02-08   # Download daily Excel report
GET /api/export/monthly?month=2&year=2026  # Download monthly Excel report
```

---

# 3. SERVICES LAYER

## 3.1 `services/face_detector.py` (305 lines)

**Purpose**: Face detection using OpenCV YuNet DNN model.

### Key Features:
- **Smoothing Algorithm**: Reduces jitter by tracking previous detections and blending positions
- **5-Point Landmarks**: Extracts eye, nose, and mouth corners
- **Async Support**: Non-blocking detection via ThreadPoolExecutor

### Core Methods:

#### `initialize()`
```python
def initialize():
    # Load YuNet ONNX model
    self._detector = cv2.FaceDetectorYN_create(
        model_path,
        "",
        (320, 320),  # Input size
        score_threshold=0.7,
        nms_threshold=0.3
    )
```

#### `detect(image)` → `List[FaceDetection]`
**Algorithm**:
```
1. Set input size to match image dimensions
2. Run YuNet inference → Returns array of detections
3. Parse each detection:
   - Extract bbox: (x, y, w, h)
   - Extract confidence: face[14]
   - Extract 5 landmarks: face[4:14]
4. Apply smoothing to reduce jitter
5. Return List[FaceDetection]
```

**Smoothing Logic** (Lines 105-160):
- Matches new detections to previous ones by proximity
- Blends positions: `smooth_x = prev_x * 0.6 + new_x * 0.4`
- Prevents sudden jumps in bounding boxes

#### `extract_face(image, detection)` → Cropped face image
Extracts face region with 20% padding for better recognition.

---

## 3.2 `services/face_recognizer.py` (309 lines)

**Purpose**: Face recognition using SFace DNN (128D embeddings).

### Architecture:
```
Input Image (112x112) 
  → SFace CNN 
  → 128D Embedding 
  → L2 Normalization 
  → Unit Vector
```

### Key Methods:

#### `_compute_embedding(face_image)` → `np.ndarray[128]`
```python
# Resize to 112x112
face_input = cv2.resize(face_image, (112, 112))

# Extract 128D feature
feature = self._recognizer.feature(face_input)

# Normalize to unit vector
norm = np.linalg.norm(feature)
feature = feature / norm
```

#### `recognize(face_image)` → `Optional[RecognitionMatch]`
**Vectorized Batch Comparison**:
```python
# Compute query embedding
query_embedding = self._compute_embedding(face_image)  # Shape: (128,)

# Compare against all enrolled embeddings at once
similarities = np.dot(self._embedding_matrix, query_embedding)  
# embedding_matrix: (N, 128), result: (N,)

# Find best match
best_idx = np.argmax(similarities)
best_score = similarities[best_idx]

# Threshold check
if best_score > 0.40:
    return RecognitionMatch(student_id, name, roll_number, best_score)
```

**Why this works**:
- All embeddings are normalized to unit vectors
- Dot product of normalized vectors = Cosine Similarity
- Single matrix operation compares against ALL students instantly
- **Performance**: O(N) instead of O(N) individual comparisons

#### `_load_enrolled_faces()` (Lines 93-156)
**Navigation**:
```
1. Fetch all students from MongoDB
2. Extract face_embeddings arrays
3. Convert to numpy arrays
4. Normalize each embedding
5. Build embedding_matrix: Stack all into (N, 128) matrix
6. Build parallel student_info list
```

---

## 3.3 `services/behavior_detector.py` (287 lines)

**Purpose**: Detect student behavior using YOLOv8-pose keypoint analysis.

### How it Works:

#### **Step 1: Pose Detection**
```python
results = self._model(image, verbose=False, conf=0.15)
keypoints_data = result.keypoints.data.cpu().numpy()
# Shape: (num_people, 17, 3) → 17 COCO keypoints per person
```

#### **Step 2: Behavior Classification** (Lines 141-265)
**Rule-Based Algorithm** using geometric analysis:

```python
def classify_behavior(keypoints):
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    # Calculate reference points
    shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    
    # SLEEPING: Head below shoulders
    if nose[1] > shoulder_mid_y + 20:
        return "sleeping", 0.9
    
    # DISTRACTED: Head turned sideways (>40% offset)
    nose_offset = abs(nose[0] - shoulder_mid_x)
    if nose_offset / shoulder_width > 0.4:
        return "distracted", 0.7
    
    # STUDYING: Hands below shoulders (writing posture)
    if left_wrist[1] > shoulder_mid_y or right_wrist[1] > shoulder_mid_y:
        return "studying", 0.5
    
    # DEFAULT: FOCUSED
    return "focused", 0.5
```

**COCO Keypoint Indices**:
```
0: Nose           9: Left Wrist
1: Left Eye      10: Right Wrist
2: Right Eye     11: Left Hip
3: Left Ear      12: Right Hip
4: Right Ear     13: Left Knee
5: Left Shoulder 14: Right Knee
6: Right Shoulder
7: Left Elbow
8: Right Elbow
```

---

## 3.4 `services/signaling.py` (155 lines)

**Purpose**: Manages WebSocket connections for real-time communication.

### Data Structures:
```python
@dataclass
class Connection:
    websocket: WebSocket
    room: str  # "gate", "classroom", or "dashboard"
    connected_at: datetime
    client_id: str

class ConnectionManager:
    rooms: Dict[str, Set[Connection]]  # Room → Connections
    connections: Dict[WebSocket, Connection]  # WebSocket → Connection
```

### Key Methods:

#### `connect(websocket, room)` → `Connection`
1. Accept WebSocket connection
2. Create Connection object with unique client_id
3. Add to appropriate room ("gate", "classroom", "dashboard")
4. Broadcast "camera_connected" event to dashboard
5. Return connection object

#### `broadcast_to_room(room, message)`
```python
for conn in self.rooms[room]:
    try:
        await conn.websocket.send_json(message)
    except Exception:
        # Mark for cleanup if send fails
```

#### `disconnect(websocket)`
1. Remove from connections dict
2. Remove from room set
3. Broadcast "camera_disconnected" to dashboard

---

## 3.5 `services/export.py` (173 lines)

**Purpose**: Generate Excel reports with color-coded attendance.

### Excel Formatting:
```python
# Colors
GREEN_FILL = PatternFill("C6EFCE")  # Present
YELLOW_FILL = PatternFill("FFEB9C")  # Late
RED_FILL = PatternFill("FFC7CE")    # Absent
```

### `generate_daily_excel(data)` → `BytesIO`
**Structure**:
```
┌─────────────────────────────────────┐
│  Daily Attendance Report - 2026-02-08 │
├─────────────────────────────────────┤
│  Enrolled: 50 | Present: 45 | Late: 3 | Absent: 5 │
├──────────┬─────────┬────────┬────────┤
│ Name     │ Roll No │ Time   │ Status │
├──────────┼─────────┼────────┼────────┤
│ John Doe │ CS001   │ 8:15 AM│ Present│ (Green)
│ Jane S.  │ CS002   │ 8:45 AM│ Late   │ (Yellow)
│ Bob M.   │ CS003   │ N/A    │ Absent │ (Red)
└──────────┴─────────┴────────┴────────┘
```

### `generate_monthly_excel(data)` → `BytesIO`
**Structure** (Grid format):
```
┌────────┬────┬───┬───┬───┬───┬───┬───┬───┬───┐
│ Name   │ 1  │ 2 │ 3 │ 4 │ 5 │...│ 28│ 29│ 30│
├────────┼────┼───┼───┼───┼───┼───┼───┼───┼───┤
│ John   │ P  │ P │ L │ P │ A │...│ P │ P │ P │
│ Jane   │ P  │ P │ P │ P │ P │...│ L │ P │ P │
└────────┴────┴───┴───┴───┴───┴───┴───┴───┴───┘
Legend: P=Present(Green), L=Late(Yellow), A=Absent(Red)
```

---

## 3.6 `services/audio_service.py` (47 lines)

**Purpose**: Placeholder for audio announcements (client-side TTS).

**How it works**:
- Server logs messages to console
- Actual speech synthesis happens in browser using Web Speech API
- WebSocket messages trigger client-side `speechSynthesis.speak()`

---

# 4. DATABASE MODULE

## 4.1 `database/mongodb.py` (885 lines)

**Purpose**: Async MongoDB operations using Motor driver.

### Collections:

#### **students**
```javascript
{
  "_id": ObjectId,
  "name": "John Doe",
  "roll_number": "CS2024001",
  "email": "john@college.edu",
  "face_embeddings": [[...128 floats...], [...], [...]],  // Max 3
  "face_image_paths": ["path1.jpg", "path2.jpg"],
  "is_active": true,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

#### **attendance**
```javascript
{
  "_id": ObjectId,
  "student_id": ObjectId,
  "date": "2026-02-08",
  "check_in_time": ISODate,
  "status": "present",  // "present", "late", "absent"
  "detection_confidence": 0.87,
  "auto_detected": true,
  "camera": "gate",
  "lecture": "Morning Session"
}
```

#### **behavior_logs**
```javascript
{
  "_id": ObjectId,
  "timestamp": ISODate,
  "date": "2026-02-08",
  "hour": 9,
  "stats": {
    "studying": 15,
    "focused": 20,
    "distracted": 8,
    "sleeping": 2
  }
}
```

### Key Functions:

#### `mark_attendance(student_id, confidence, status, lecture)`
**Navigation**:
```
1. Get today's date
2. Check duplicate:
   query = {student_id, date, lecture}
3. If exists:
   - If manual override (auto_detected=False), update status
   - Else return None (already marked)
4. If new:
   - Insert attendance record
   - Return inserted document
```

#### `get_attendance_trend(days=7)` → `List[Dict]`
**MongoDB Aggregation Pipeline**:
```python
[
  {$match: {date: {$gte: "2026-02-01", $lte: "2026-02-07"}}},
  {$group: {
    _id: "$date",
    present: {$sum: 1}
  }},
  {$sort: {_id: 1}}
]
```

Returns:
```python
[
  {"date": "2026-02-01", "present": 45, "absent": 5, "total": 50},
  {"date": "2026-02-02", "present": 47, "absent": 3, "total": 50},
  ...
]
```

#### `get_current_lecture_status()` → `(lecture_name, status)`
**Logic**:
```
1. Load settings.lectures from MongoDB
2. Get current time in minutes (e.g., 9:45 AM = 585 minutes)
3. For each lecture:
   - Check if current_time within (start - 30min, end)
   - Calculate lateness:
     * Before start + late_buffer → "present"
     * After start + late_buffer → "late"
4. Return (lecture_name, status) or (None, "present")
```

---

# 5. FRONTEND (JavaScript)

## 5.1 `static/js/dashboard.js` (361 lines)

**Purpose**: Dashboard WebSocket client and UI updates.

### Class Structure:
```javascript
class Dashboard {
  constructor() {
    this.ws = null;
    this.reconnectAttempts = 0;
  }
  
  init() {
    this.connectWebSocket();
    this.loadStats();
    this.loadAttendance();
  }
}
```

### Message Handlers:

#### `handleGateSnapshot(data)`
```javascript
// Receives annotated frame from server
const img = container.querySelector('img');
img.src = `data:image/jpeg;base64,${data.image}`;
```

#### `handleClassroomSnapshot(data)`
```javascript
// Update classroom feed
img.src = `data:image/jpeg;base64,${data.image}`;

// Update behavior stats
updateBehaviorStats({
  studying: data.stats.studying,
  focused: data.stats.focused,
  distracted: data.stats.distracted,
  sleeping: data.stats.sleeping
});
```

#### `loadAttendance()`
```javascript
// Fetch from API
const records = await fetch('/api/attendance/today').then(r => r.json());

// Render table
tbody.innerHTML = records.map(r => `
  <tr>
    <td>${r.name}</td>
    <td>${r.roll_number}</td>
    <td>${new Date(r.entry_time).toLocaleTimeString()}</td>
    <td><span class="badge badge-success">${r.status}</span></td>
    <td>${(r.confidence * 100).toFixed(0)}%</td>
  </tr>
`).join('');
```

---

## 5.2 `static/js/webrtc-client.js` (349 lines)

**Purpose**: Camera client for gate/classroom phones.

### Workflow:

#### **1. Camera Initialization**
```javascript
async start(videoElement) {
  // Request camera access
  this.localStream = await navigator.mediaDevices.getUserMedia({
    video: {width: 640, height: 480, frameRate: 30},
    audio: true
  });
  
  // Display local preview
  videoElement.srcObject = this.localStream;
  
  // Connect to server
  await this.connectWebSocket();
  
  // Start sending frames
  this.startFrameSending();
}
```

#### **2. Frame Capture Loop**
```javascript
startFrameSending() {
  const interval = this.room === 'classroom' ? 10000 : 66;  // 10s vs 66ms
  const quality = this.room === 'classroom' ? 0.9 : 0.6;
  
  this.frameInterval = setInterval(() => {
    // Draw video to canvas
    canvas.width = 640;
    canvas.height = 480;
    ctx.drawImage(this.videoElement, 0, 0, 640, 480);
    
    // Encode to JPEG
    const dataUrl = canvas.toDataURL('image/jpeg', quality);
    
    // Send via WebSocket
    this.ws.send(JSON.stringify({
      type: 'process_frame',  // or 'classroom_snapshot'
      frame: dataUrl,
      timestamp: new Date().toISOString()
    }));
  }, interval);
}
```

**Gate vs Classroom**:
- **Gate**: 66ms interval (15 FPS), 60% quality, 640x480
- **Classroom**: 10s interval, 90% quality, up to 1280px width

#### **3. Receive Annotated Frames**
```javascript
handleMessage(message) {
  switch (message.type) {
    case 'gate_snapshot':
      // Server sends back annotated frame
      this.onAnnotatedFrame(message.image, message.face_count);
      break;
    
    case 'new_attendance':
      this.onAttendanceMarked(message);
      break;
  }
}
```

---

## 5.3 `static/js/webrtc-viewer.js` (222 lines)

**Purpose**: Dashboard WebRTC viewer (P2P streaming - currently unused).

**Note**: The current system uses WebSocket frame transmission instead of P2P WebRTC. This file provides infrastructure for future P2P implementation.

---

# 6. FRONTEND (CSS)

## 6.1 `static/css/modern.css` (16KB)

**Purpose**: Modern UI styling with glassmorphism and animations.

**Key Features**:
- CSS Variables for theming
- Gradient backgrounds
- Glassmorphism effects (backdrop-filter)
- Smooth animations
- Responsive grid layouts

---

## 6.2 `static/css/styles.css` (12KB)

**Purpose**: Base styles and utility classes.

---

# 7. TEMPLATES (HTML)

## 7.1 `templates/dashboard_new.html`

**Structure**:
```html
<div class="dashboard-grid">
  <!-- Stats Cards -->
  <div class="stats-row">
    <div class="stat-card">Total Students: <span id="total-students"></span></div>
    <div class="stat-card">Present Today: <span id="present-today"></span></div>
  </div>
  
  <!-- Camera Feeds -->
  <div class="camera-section">
    <div class="feed-container" id="gate-feed">
      <img />  <!-- Receives base64 frames via WebSocket -->
    </div>
    <div class="feed-container" id="classroom-feed">
      <img id="classroom-snapshot" />
    </div>
  </div>
  
  <!-- Behavior Stats -->
  <div class="behavior-stats">
    <span id="stat-studying">📖 0</span>
    <span id="stat-focused">🎯 0</span>
    <span id="stat-distracted">👀 0</span>
    <span id="stat-sleeping">😴 0</span>
  </div>
  
  <!-- Attendance Table -->
  <table>
    <tbody id="attendance-list"></tbody>
  </table>
</div>

<script src="/static/js/dashboard.js"></script>
```

---

## 7.2 `templates/gate_camera.html`

**Structure**:
```html
<video id="video" autoplay playsinline></video>
<canvas id="overlay"></canvas>  <!-- For drawing face boxes -->

<script src="/static/js/webrtc-client.js"></script>
<script>
  const client = new WebRTCClient({
    room: 'gate',
    onAnnotatedFrame: (image, faceCount) => {
      // Display server-rendered frame with boxes
    },
    onAttendanceMarked: (data) => {
      showToast(`✅ ${data.name} marked present!`);
    }
  });
  
  client.start(document.getElementById('video'));
</script>
```

---

## 7.3 `templates/classroom_camera.html`

Similar to gate camera but:
- Sends frames at 10-second intervals
- Higher quality (90% JPEG)
- Processes both faces and behaviors

---

## 7.4 `templates/add_students.html`

**3-Photo Enrollment System**:
```html
<form id="enrollment-form">
  <input type="text" name="name" required>
  <input type="text" name="roll_number" required>
  <input type="email" name="email">
  
  <!-- Camera capture for 3 photos -->
  <video id="camera"></video>
  <button onclick="capturePhoto(1)">Capture Photo 1</button>
  <button onclick="capturePhoto(2)">Capture Photo 2</button>
  <button onclick="capturePhoto(3)">Capture Photo 3</button>
  
  <img id="preview-1" />
  <img id="preview-2" />
  <img id="preview-3" />
  
  <button type="submit">Enroll Student</button>
</form>

<script>
async function capturePhoto(step) {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 480;
  canvas.getContext('2d').drawImage(video, 0, 0);
  
  const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
  
  // Send to server
  await fetch(`/api/mongodb/students/${studentId}/enroll-face`, {
    method: 'POST',
    body: JSON.stringify({step, face_image: dataUrl})
  });
}
</script>
```

**Server-side processing**:
1. Receives 3 face images
2. Extracts embedding from each
3. **Averages the 3 embeddings** for better accuracy
4. Normalizes the averaged embedding
5. Stores in MongoDB

---

## 7.5 `templates/analysis.html`

**Charts**:
- Attendance Trend (7 days) - Line chart
- Behavior Distribution - Pie chart
- Attendance Rate - Progress bar

Uses Chart.js library.

---

## 7.6 `templates/data.html`

**Features**:
- Student list with search/filter
- Attendance history grid (last 5 days)
- Color-coded status indicators

---

## 7.7 `templates/settings.html`

**Lecture Schedule Configuration**:
```javascript
lectures: [
  {
    name: "Morning Session",
    start_time: "08:30",
    end_time: "12:00",
    late_buffer: 15  // minutes
  }
]
```

---

# 8. DEBUG SCRIPTS

## 8.1 `debug_behavior.py`

**Purpose**: Test behavior detection on a static image.

**Usage**:
```bash
python debug_behavior.py
```

**What it does**:
1. Loads test image
2. Runs YOLOv8-pose detection
3. Prints detected behaviors and confidences

---

## 8.2 `debug_cv2.py`

**Purpose**: Verify OpenCV installation and locate FaceRecognizerSF.

**Output**:
```
OpenCV Version: 4.8.1
cv2.face attributes: [...]
Found cv2.FaceRecognizerSF_create ✅
```

---

## 8.3 `debug_sface.py`

**Purpose**: Test SFace embedding extraction on dummy image.

**What it does**:
1. Creates 112x112 test image
2. Loads SFace model
3. Computes embedding
4. Prints shape and sample values

---

# 9. UTILITY SCRIPTS

## 9.1 `download_models.py`

**Purpose**: Download required AI models.

**What it downloads**:
1. **YuNet** (233 KB) - Face detection
   - URL: `opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
   
2. **SFace** (37 MB) - Face recognition
   - URL: `opencv_zoo/models/face_recognition_sface/face_recognition_sface_2021dec.onnx`

**Usage**:
```bash
python download_models.py
```

---

## 9.2 `verify_recognition.py`

**Purpose**: Test face recognition pipeline end-to-end.

**What it tests**:
1. MongoDB connection
2. Face recognizer initialization
3. Embedding loading from database
4. Recognition on test image from `data/students/`

**Usage**:
```bash
python verify_recognition.py
```

---

# 10. OTHER FILES

## 10.1 `yolov8n-pose.pt` (6.5 MB)

**Purpose**: Pre-trained YOLOv8-nano pose estimation model.

**Format**: PyTorch weights file (.pt)

**How it's used**:
```python
from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
results = model(image)
keypoints = results[0].keypoints.data  # Shape: (N, 17, 3)
```

---

## 10.2 `LICENSE`

**Type**: MIT License
**Copyright**: 2026 Jeet Patel

---

## 10.3 `README.md`

**Contents**:
- Project overview
- Installation instructions
- API reference
- Configuration guide

---

## 10.4 `WALKTHROUGH.md`

**Contents**:
- Step-by-step demonstration guide
- Troubleshooting common issues
- Expected behavior for each scenario

---

# 11. DOWNLOADED MODULES - DETAILED EXPLANATION

## FastAPI & Uvicorn
- **FastAPI**: Modern web framework with automatic API documentation
- **Uvicorn**: ASGI server that runs FastAPI apps
- **Usage**: Handles HTTP routes, WebSocket connections, dependency injection

## OpenCV (opencv-contrib-python)
- **YuNet DNN**: Deep neural network for face detection
- **SFace DNN**: Face recognition using 128D embeddings
- **Functions**: `cv2.FaceDetectorYN_create`, `cv2.FaceRecognizerSF_create`

## Ultralytics (YOLOv8)
- **Purpose**: Real-time pose estimation
- **Output**: 17 COCO keypoints per person
- **Usage**: Detect body posture for behavior classification

## Motor (MongoDB Driver)
- **Purpose**: Async MongoDB operations
- **Benefits**: Non-blocking database queries in async FastAPI
- **Usage**: `AsyncIOMotorClient`, await-able queries

## NumPy
- **Purpose**: Matrix operations for embeddings
- **Key Operations**:
  - `np.dot()` for cosine similarity
  - `np.linalg.norm()` for normalization
  - `np.vstack()` for embedding matrix construction

## WebSockets
- **Purpose**: Real-time bidirectional communication
- **Usage**: Send frames from camera, receive results on dashboard
- **Advantage**: Lower latency than HTTP polling

## Python-Multipart
- **Purpose**: Parse form data with file uploads
- **Usage**: Student enrollment with face images

## Aiofiles
- **Purpose**: Async file I/O
- **Usage**: Save student photos without blocking

## Jinja2
- **Purpose**: HTML template rendering
- **Usage**: Server-side rendering of dashboard, camera pages

## Python-Dotenv
- **Purpose**: Load environment variables from `.env`
- **Usage**: Secure MongoDB connection string storage

---

# 12. COMPLETE SYSTEM FLOW

## Attendance Marking Flow (Gate Camera)

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Mobile Camera (Gate)                                            │
│    - getUserMedia() captures video                                 │
│    - Every 66ms: Draw frame to canvas → JPEG → base64             │
│    - WebSocket.send({type: "process_frame", frame: "..."})        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│ 2. Server (app.py - process_frame)                                 │
│    - Decode base64 → numpy array                                   │
│    - face_detector.detect(frame) → List[FaceDetection]            │
│    - For each face:                                                │
│      * face_detector.extract_face() → crop                        │
│      * face_recognizer.recognize(crop) → RecognitionMatch         │
│      * If match: mongodb.mark_attendance(student_id)              │
│    - face_detector.draw_detections() → annotated frame            │
│    - Encode to base64                                             │
│    - WebSocket.broadcast({type: "gate_snapshot", image: "..."})  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│ 3. Dashboard (dashboard.js)                                        │
│    - handleGateSnapshot(data)                                      │
│    - Display annotated image                                       │
│    - loadStats() → Update "Present Today" count                   │
│    - showToast("✅ Student marked present!")                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Behavior Detection Flow (Classroom Camera)

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Mobile Camera (Classroom)                                       │
│    - Every 10 seconds: Capture high-quality frame (90% JPEG)      │
│    - WebSocket.send({type: "classroom_snapshot", frame: "..."})   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│ 2. Server (app.py - process_classroom_snapshot)                   │
│    Step 1: Face Recognition                                        │
│      - face_detector.detect() → faces                             │
│      - For each: face_recognizer.recognize() → names              │
│                                                                    │
│    Step 2: Behavior Detection                                      │
│      - behavior_detector.detect() → List[BehaviorDetection]       │
│      - Each detection has: behavior, confidence, bbox             │
│                                                                    │
│    Step 3: Annotation                                              │
│      - Draw green boxes for faces + names                         │
│      - Draw colored boxes for behaviors:                          │
│        * Studying=Green, Focused=Blue, Distracted=Orange, etc.   │
│                                                                    │
│    Step 4: Broadcast                                               │
│      - Send annotated frame to dashboard                          │
│      - mongodb.log_classroom_snapshot(stats) [rate-limited 10s]  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│ 3. Dashboard                                                        │
│    - Display annotated classroom feed                              │
│    - Update behavior stats: 📖15 🎯20 👀8 😴2                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

# 13. KEY ALGORITHMS EXPLAINED

## Face Recognition - Cosine Similarity

**Why it works**:
```
Given two unit vectors A and B:
Cosine Similarity = A · B = cos(θ)

If θ = 0° (same direction) → cos(0) = 1.0 → Same person
If θ = 90° (perpendicular) → cos(90) = 0.0 → Different people
If θ < 66.4° → cos(θ) > 0.40 → Match threshold
```

**Implementation**:
```python
# All embeddings normalized to unit length
embeddings_matrix = np.array([
  [0.1, 0.2, ..., 0.8],  # Student 1
  [0.3, -0.1, ..., 0.5], # Student 2
  ...
])  # Shape: (N_students, 128)

query = np.array([0.2, 0.15, ..., 0.7])  # Shape: (128,)

# Vectorized comparison (single operation)
similarities = embeddings_matrix @ query  # Shape: (N_students,)

best_match_idx = np.argmax(similarities)
if similarities[best_match_idx] > 0.40:
    # Match found
```

---

## Behavior Detection - Geometric Rules

**Sleeping Detection**:
```python
# If nose (head) is below shoulder line → Head on desk
if nose_y > shoulder_mid_y + 20:
    return "sleeping"
```

**Distracted Detection**:
```python
# If head offset > 40% of shoulder width → Turned away
nose_offset = abs(nose_x - shoulder_center_x)
if nose_offset / shoulder_width > 0.4:
    return "distracted"
```

**Studying Detection**:
```python
# If hands below shoulders → Writing posture
if left_wrist_y > shoulder_y or right_wrist_y > shoulder_y:
    return "studying"
```

---

## Smoothing Algorithm (Face Detection)

**Problem**: Face bounding boxes jitter between frames.

**Solution**: Exponential smoothing
```python
# Match new detection to previous by proximity
prev_bbox = find_closest_previous(new_bbox)

# Blend positions (60% old, 40% new)
smooth_x = prev_x * 0.6 + new_x * 0.4
smooth_y = prev_y * 0.6 + new_y * 0.4
smooth_w = prev_w * 0.6 + new_w * 0.4
smooth_h = prev_h * 0.6 + new_h * 0.4
```

**Result**: Stable bounding boxes without sudden jumps.

---

# 14. NAVIGATION MAP

## How to Navigate the Codebase

### **To understand attendance flow**:
1. Start: `templates/gate_camera.html` → Camera capture
2. Follow: `static/js/webrtc-client.js` → Frame sending
3. Then: `app.py` → `process_frame()` → Face recognition
4. Finally: `database/mongodb.py` → `mark_attendance()`

### **To understand behavior detection**:
1. Start: `templates/classroom_camera.html`
2. Follow: `app.py` → `process_classroom_snapshot()`
3. Deep dive: `services/behavior_detector.py` → `_classify_behavior()`

### **To understand face recognition**:
1. Model: `services/face_recognizer.py` → `_compute_embedding()`
2. Comparison: `recognize()` → Vectorized cosine similarity
3. Storage: `database/mongodb.py` → `add_face_embedding()`

### **To understand frontend**:
1. Layout: `templates/dashboard_new.html`
2. Logic: `static/js/dashboard.js`
3. Styling: `static/css/modern.css`

---

# 15. CONFIGURATION & SETUP

## Environment Variables (.env)
```env
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority
```

## First-Time Setup
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download AI models
python download_models.py

# 4. Configure database
# Create .env file with MONGODB_URI

# 5. Run server
python app.py
```

## Access Points
- **Server**: http://localhost:8000
- **Dashboard**: http://localhost:8000/dashboard
- **Gate Camera**: http://[YOUR-IP]:8000/gate (on mobile)
- **Classroom Camera**: http://[YOUR-IP]:8000/classroom (on mobile)

---

# END OF DOCUMENTATION

This documentation covers every file in the EYE-X project with detailed explanations of how each component works, navigates, and integrates with other parts of the system.
