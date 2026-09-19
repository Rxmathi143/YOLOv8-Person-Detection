# 🎯 Face Attendance System

### 🔐 AI-Powered Face Recognition Attendance System using Python & OpenCV

An automated **face recognition-based attendance system** built with **Python, OpenCV, YuNet, and SFace**.

The system identifies registered **Admin and Candidates** through a webcam and automatically records their **Check-In, Check-Out, and Total Duration** in a CSV attendance file.

---

## ✨ Project Overview

This project provides an automated attendance solution using **face detection and face recognition**.

Instead of manually marking attendance, the system recognizes a person's face and manages their attendance automatically.

### 🔄 How it works

```text
                 ┌──────────────────────┐
                 │      START SYSTEM    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     Open Webcam      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Detect Face       │
                 │       YuNet          │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Recognize Person    │
                 │       SFace          │
                 └──────────┬───────────┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
            ┌─────────┐          ┌────────────┐
            │ UNKNOWN │          │ REGISTERED │
            └─────────┘          └─────┬──────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ CHECK-IN TIMER  │
                              │     3 Seconds   │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │    CHECKED IN   │
                              └────────┬────────┘
                                       │
                                Person Leaves
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  Leave Detected │
                              │     1 Second    │
                              └────────┬────────┘
                                       │
                                Person Returns
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ CHECK-OUT TIMER │
                              │     3 Seconds   │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   CHECKED OUT   │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ Attendance CSV  │
                              │    Updated      │
                              └─────────────────┘
```

---

# 🚀 Features

### 👤 Admin Attendance

* Admin face is registered separately.
* Admin can also check in and check out.
* Admin is identified automatically through face recognition.
* Admin attendance is stored in the same attendance system.

### 👨‍💼 Candidate Registration

* Admin verification is required before registering a candidate.
* Candidate face is automatically captured.
* Candidate name is entered using an on-screen keyboard.
* Each candidate receives a unique ID.
* Face features are stored as `.npy` files.
* Candidate information is stored in `candidates.json`.

### 🧠 Face Detection

Uses **YuNet Face Detection** to detect faces from the webcam.

### 🔍 Face Recognition

Uses **SFace Face Recognition** to compare detected faces with registered face features.

### ⏱️ Automatic Check-In

A registered person must remain visible for:

```text
3 seconds
```

The camera displays a live timer:

```text
CHECK IN: 1.8 / 3.0s
```

After 3 seconds:

```text
CHECKED IN
```

### 🚪 Automatic Leave Detection

After a person disappears from the camera for:

```text
1 second
```

the system considers that the person has left.

### 🔄 Automatic Check-Out

When the person returns, they must remain visible for:

```text
3 seconds
```

The camera displays:

```text
CHECK OUT: 2.1 / 3.0s
```

After 3 seconds, checkout is completed automatically.

### 🧮 Automatic Duration Calculation

The system calculates the total time between:

```text
Check In → Check Out
```

Example:

```text
5h 24m 18s
```

### 📊 CSV Attendance

Attendance is automatically stored in:

```text
output/attendance.csv
```

Example:

| Date       | ID | Name         | Check In | Check Out | Duration   | Status    |
| ---------- | -: | ------------ | -------- | --------- | ---------- | --------- |
| 2026-09-19 |  1 | ADMIN        | 09:10:25 | 17:15:42  | 8h 05m 17s | Completed |
| 2026-09-19 |  2 | Harin Venkat | 09:15:10 | 16:50:22  | 7h 35m 12s | Completed |

---

# 🛠️ Technologies Used

| Technology | Purpose                               |
| ---------- | ------------------------------------- |
| 🐍 Python  | Main programming language             |
| 👁️ OpenCV | Computer vision and camera processing |
| 🧠 YuNet   | Face detection                        |
| 🔍 SFace   | Face recognition                      |
| 🪟 Tkinter | Graphical user interface              |
| 🖼️ Pillow | Displaying camera frames              |
| 🔢 NumPy   | Face feature storage and processing   |
| 📄 JSON    | Candidate information storage         |
| 📊 CSV     | Attendance data storage               |

---

# 📁 Project Structure

```text
yolo person detection/
│
├── 📄 main.py
├── 📄 admin_register.py
├── 📄 register_candidate.py
├── 📄 register_face.py
├── 📄 detect_image.py
├── 📄 yolov8n.pt
│
├── 📁 models/
│   ├── face_detection_yunet_2026may.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── 📁 faces/
│   ├── admin.jpg
│   ├── admin.npy
│   │
│   └── 📁 candidates/
│       ├── candidate_2.npy
│       └── candidates.json
│
└── 📁 output/
    └── attendance.csv
```

---

# 🔐 Registration System

## 1️⃣ Admin Registration

The Admin must first register their face.

Run:

```bash
python admin_register.py
```

The system captures the Admin's face and generates:

```text
faces/admin.jpg
faces/admin.npy
```

The `.npy` file contains the numerical face feature used for recognition.

---

## 2️⃣ Candidate Registration

After Admin registration, candidates can be registered.

Run:

```bash
python register_candidate.py
```

The process is:

```text
Admin Verification
       ↓
Candidate Face Detection
       ↓
Automatic Face Capture
       ↓
Enter Candidate Name
       ↓
Generate Candidate ID
       ↓
Save Face Feature
       ↓
Update candidates.json
```

Example:

```text
Candidate ID   : 2
Candidate Name : Harin Venkat
Feature File   : candidate_2.npy
```

---

# 🎥 Attendance System

Start the main application:

```bash
python main.py
```

The webcam opens automatically.

The system continuously:

```text
Detect Face
     ↓
Extract Face Feature
     ↓
Compare With Registered Faces
     ↓
Identify Person
     ↓
Start Attendance Timer
```

---

# ⏱️ Attendance Logic

## 🟡 Check-In

When a registered person enters the camera:

```text
CHECK IN: 0.0 / 3.0s
```

The timer continues:

```text
CHECK IN: 1.0 / 3.0s
CHECK IN: 2.0 / 3.0s
CHECK IN: 3.0 / 3.0s
```

Then:

```text
✅ CHECKED IN
```

---

## 🟢 Person Stays Inside

While the person remains visible:

```text
CHECKED IN
ATTENDANCE ACTIVE
```

The check-in time is stored in the CSV file.

---

## 🟠 Person Leaves

When the face disappears:

```text
Leave Detection
      ↓
Wait 1 Second
      ↓
Checkout Pending
```

The system does not immediately check out the person.

This helps prevent accidental checkout caused by temporary face detection loss.

---

## 🔵 Check-Out

When the person returns:

```text
CHECK OUT: 0.0 / 3.0s
```

The timer starts again.

After 3 seconds:

```text
✅ CHECKED OUT
```

The system calculates:

```text
Duration = Check Out Time - Check In Time
```

---

# 🧠 Face Recognition

The project uses **SFace** to generate a numerical representation of each face.

During registration:

```text
Face
 ↓
SFace
 ↓
Face Feature
 ↓
.npy file
```

During attendance:

```text
Webcam Face
 ↓
SFace
 ↓
Face Feature
 ↓
Compare With Registered Features
 ↓
Best Match
```

The project uses a cosine similarity threshold:

```python
FACE_MATCH_THRESHOLD = 0.45
```

A face is considered recognized when the similarity score reaches the configured threshold.

---

# 📊 Attendance CSV

The generated CSV contains:

```text
Date
ID
Name
Check In
Check Out
Duration
Status
```

Example:

```csv
Date,ID,Name,Check In,Check Out,Duration,Status
2026-09-19,1,ADMIN,09:10:25,17:15:42,8h 05m 17s,Completed
2026-09-19,2,Harin Venkat,09:15:10,16:50:22,7h 35m 12s,Completed
```

---

# ⚙️ Configuration

The main attendance settings can be changed inside `main.py`.

```python
FACE_MATCH_THRESHOLD = 0.45

CHECK_IN_SECONDS = 3.0
CHECK_OUT_SECONDS = 3.0

LEAVE_SECONDS = 1.0
```

### Meaning

| Setting                |  Value | Purpose                         |
| ---------------------- | -----: | ------------------------------- |
| `FACE_MATCH_THRESHOLD` | `0.45` | Face recognition threshold      |
| `CHECK_IN_SECONDS`     |  `3.0` | Time required for check-in      |
| `CHECK_OUT_SECONDS`    |  `3.0` | Time required for checkout      |
| `LEAVE_SECONDS`        |  `1.0` | Time required to detect leaving |

---

# 📦 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Rxmathi143/YOLOv8-Person-Detection.git
```

Move into the project:

```bash
cd YOLOv8-Person-Detection
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install opencv-python numpy pillow ultralytics
```

---

# ▶️ Running the Project

### Step 1 — Register Admin

```bash
python admin_register.py
```

### Step 2 — Register Candidates

```bash
python register_candidate.py
```

### Step 3 — Start Attendance

```bash
python main.py
```

---

# 🖥️ System Requirements

### Hardware

* 💻 Windows computer
* 📷 Webcam
* 🧠 Recommended: modern Intel/AMD processor
* 💾 Minimum 4 GB RAM
* 💾 Recommended 8 GB+ RAM

### Software

* Python 3.10+
* OpenCV
* NumPy
* Pillow
* Tkinter
* Ultralytics

---

# 🔒 Security & Recognition

The project does not store only the person's name for recognition.

Instead, the system stores a numerical face feature:

```text
Face Image
     ↓
SFace
     ↓
Numerical Feature
     ↓
.npy
```

The recognition system compares the live camera feature against registered features.

---

# 🎯 Future Improvements

The project can be extended with:

* 📱 Android application
* 🌐 Web-based attendance dashboard
* ☁️ Cloud database
* 📧 Automatic attendance emails
* 📈 Attendance analytics
* 📅 Monthly attendance reports
* 🗓️ Calendar-based attendance
* 👥 Larger-scale employee management
* 🔐 Secure admin login
* 🗄️ MySQL/PostgreSQL/Firebase database
* 📊 Graphs and attendance statistics
* 📤 Excel report generation
* 🖥️ Real-time admin dashboard

---

# 🧩 Project Workflow

```text
                    FACE ATTENDANCE SYSTEM
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
       ADMIN REGISTER                    CANDIDATE REGISTER
             │                                 │
             ▼                                 ▼
       admin.npy                        candidate_X.npy
             │                                 │
             └────────────────┬────────────────┘
                              │
                              ▼
                         main.py
                              │
                              ▼
                           WEBCAM
                              │
                              ▼
                         YuNet
                     Face Detection
                              │
                              ▼
                          SFace
                     Face Recognition
                              │
                              ▼
                     Registered Person
                              │
                     ┌────────┴────────┐
                     │                 │
                     ▼                 ▼
                 CHECK IN          CHECK OUT
                  3 sec               3 sec
                     │                 │
                     └────────┬────────┘
                              ▼
                       Duration Calculate
                              │
                              ▼
                     attendance.csv
```

---

# 💡 Why This Project?

Traditional attendance systems can require:

* Manual attendance
* ID cards
* Fingerprint devices
* Touch-based systems

This project demonstrates how **computer vision and face recognition** can be used to automate attendance.

It also demonstrates practical implementation of:

```text
Python
   +
OpenCV
   +
Face Detection
   +
Face Recognition
   +
GUI
   +
File Management
   +
Real-Time Processing
   +
Attendance Automation
```

---

# 👨‍💻 Author

### Mathivishnu S

**BTech — Information Science & Engineering**

Interested in:

```text
Python
Backend Development
Frontend Development
Computer Vision
Artificial Intelligence
Software Development
```

---

# ⭐ Project Highlights

```text
✅ Real-time face detection
✅ Real-time face recognition
✅ Admin registration
✅ Candidate registration
✅ Multiple candidate support
✅ Automatic check-in
✅ Automatic check-out
✅ Leave detection
✅ Live attendance timer
✅ Duration calculation
✅ CSV attendance records
✅ Tkinter GUI
```

---

## ⭐ If you found this project useful

Give the repository a ⭐ on GitHub!

**Built with Python, OpenCV & AI 🤖**

```

This README is designed to make the project understandable to someone who opens your GitHub repository for the first time, while also showing the **actual technical workflow** you implemented.
```
