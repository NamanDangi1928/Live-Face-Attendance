# Live Face Recognition Attendance System

## 1. Overview
Live Face Recognition Attendance System is a distributed attendance and behavioral analytics system designed for educational environments. By leveraging an "Edge-First" architecture, the system performs computationally intensive tasks (face detection and pose estimation) on client devices (smartphones and laptops), significantly reducing server bandwidth requirements and latency.
The Live Face Recognition Attendance System is an AI-powered attendance management application developed to automate student attendance using real-time face detection and recognition.

The system uses a camera to capture live video, detect human faces, and identify registered students by comparing their facial features with previously enrolled records. When a registered student is recognized, the system can record their attendance and update the attendance dashboard.

The application integrates computer vision, machine learning and database management to provide a practical and efficient alternative to traditional manual attendance methods.

High-performance WebSocket protocols and Peer-to-Peer (WebRTC) streaming enable real-time monitoring with sub-200ms latency, offering a substantial improvement over traditional RTMP-based solutions.

## 2. Key Features
Real-Time Face Detection: Detects human faces through a live webcam feed using the YuNet face detection model.

Face Recognition: Uses the SFace recognition model to identify registered students.

Student Registration: Allows student information to be registered along with multiple face photographs.

Attendance Management: Records attendance based on recognized student identities.

MongoDB Integration: Stores student information and attendance-related records in a MongoDB database.

Attendance Dashboard: Displays attendance statistics, including Present Today, Late Arrivals and Absent students.

Unknown Face Detection: Identifies faces that do not match registered students.

Camera Integration: Supports live camera input for face detection and recognition.

Improved Recognition: Recognition functionality was tested and corrected to improve identification of enrolled students.

## 3. System Architecture and Working
The application follows a client-server architecture in which the camera provides live video input, the backend processes facial information, and MongoDB manages the application data.

Working Process

Student Enrollment: Student details are entered through the registration interface, and face photographs are captured.

Face Detection: The camera captures live video, and the YuNet model detects faces in each frame.

Face Recognition: The SFace model extracts facial features and compares them with the enrolled student information.

Identity Verification: The system determines whether the detected face belongs to a registered student.

Attendance Recording: When a registered student is recognized, the attendance information is processed and recorded.

Unknown Face Identification: Faces that cannot be matched with registered students are classified as unknown.

Dashboard Update: The attendance dashboard displays the available student and attendance statistics.

## 4. Project Structure
The project is organized into different modules for application management, face detection, recognition, database operations and frontend presentation.

Live-Face-Recognition-Attendance-System/
│
├── app.py
├── face detector.py
├── recognized/
│
├── database/
│
├── models/
│
├── static/
│
├── templates/
│
├── requirements.txt
├── config.py
├── README.md
└── .gitignore

## 5. Installation & Setup

### Prerequisites
*   Python 3.9 or higher
*   MongoDB instance (Local or Atlas)
*   Webcam (for testing)

### Steps
1.  Step 1: Clone the Repository

git clone https://github.com/NamanDangi1928/Live-Face-Attendance.git

Navigate to the project directory:

cd Live-Face-Recognition-Attendance-System

Step 2: Create a Virtual Environment

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Step 3: Install Dependencies

pip install -r requirements.txt

Step 4: Configure MongoDB

Configure the MongoDB connection according to the application's database configuration.

If using MongoDB Atlas, provide the appropriate connection string and ensure that database access is configured correctly.

Do not upload database passwords or private credentials to a public GitHub repository.

Step 5: Download Required Models

Ensure that the required YuNet and SFace model files are available in the appropriate model directory.

Step 6: Run the Application

Start the application using:

python app.py

The application can then be accessed through the local server address displayed in the terminal.

## 6. DataBase Management
MongoDB is used to manage student information and attendance-related data.

The database supports the application's student enrollment and attendance management functionality.

Depending on the configured database structure, records can be inspected through MongoDB Atlas or MongoDB Compass.
## 7. Development and Improvements
During the development of this project, several modifications and improvements were made to the existing application.

Configured the Python environment and application dependencies.

Set up the application locally and tested its dashboard.

Configured and tested the YuNet and SFace recognition models.

Investigated and corrected face recognition issues.

Tested student enrollment and facial recognition using a laptop webcam.

Integrated and verified the MongoDB Atlas connection.

Tested attendance dashboard counters and their updates.

Examined unknown-face identification functionality.

Worked on improving the application for practical classroom attendance use.

