import cv2
import os
import json
import time
import csv
from datetime import datetime

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


FACE_DETECTOR = "models/face_detection_yunet_2026may.onnx"
FACE_RECOGNIZER = "models/face_recognition_sface_2021dec.onnx"

ADMIN_FEATURE = "faces/admin.npy"

CANDIDATE_FOLDER = "faces/candidates"
CANDIDATE_DATA = "faces/candidates/candidates.json"

OUTPUT_FOLDER = "output"
ATTENDANCE_FILE = "output/attendance.csv"


FACE_MATCH_THRESHOLD = 0.45

CHECK_IN_SECONDS = 3.0
CHECK_OUT_SECONDS = 3.0
LEAVE_SECONDS = 1.0

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720


ADMIN_ID = 1
ADMIN_NAME = "ADMIN"


# Check required files
if not os.path.exists(FACE_DETECTOR):
    print("ERROR: YuNet model not found.")
    print(f"Expected: {FACE_DETECTOR}")
    exit()

if not os.path.exists(FACE_RECOGNIZER):
    print("ERROR: SFace model not found.")
    print(f"Expected: {FACE_RECOGNIZER}")
    exit()

if not os.path.exists(ADMIN_FEATURE):
    print("ERROR: Admin face is not registered.")
    print("Run:")
    print("python admin_register.py")
    exit()

if not os.path.exists(CANDIDATE_DATA):
    print("ERROR: No candidate registration data found.")
    print()
    print("Run:")
    print("python register_candidate.py")
    exit()


os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Load registered admin feature
admin_feature = np.load(ADMIN_FEATURE)


# Load candidate registration data
try:
    with open(
        CANDIDATE_DATA,
        "r",
        encoding="utf-8"
    ) as file:
        candidates = json.load(file)

except Exception as error:
    print("ERROR: Could not load candidates.json")
    print(error)
    exit()


# Load candidate face features
candidate_features = []

print()
print("==========================================")
print("LOADING CANDIDATES")
print("==========================================")

for candidate in candidates:
    try:
        candidate_id = int(candidate["id"])
        candidate_name = candidate["name"]
        feature_filename = candidate["feature"]

        feature_path = os.path.join(
            CANDIDATE_FOLDER,
            feature_filename
        )

        if not os.path.exists(feature_path):
            print(
                f"WARNING: Feature file missing for "
                f"{candidate_name}"
            )
            continue

        feature = np.load(feature_path)

        candidate_features.append(
            {
                "id": candidate_id,
                "name": candidate_name,
                "feature": feature
            }
        )

        print(
            f"Loaded: ID {candidate_id} - "
            f"{candidate_name}"
        )

    except Exception as error:
        print("Could not load candidate:", error)


print("==========================================")
print(
    f"Total candidates loaded: "
    f"{len(candidate_features)}"
)
print("==========================================")
print()


# Load OpenCV face models
detector = cv2.FaceDetectorYN.create(
    FACE_DETECTOR,
    "",
    (320, 320),
    0.85,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    FACE_RECOGNIZER,
    ""
)


camera = None
camera_running = False


# Build the attendance list
attendance_people = []

attendance_people.append(
    {
        "id": ADMIN_ID,
        "name": ADMIN_NAME,
        "feature": admin_feature,
        "is_admin": True
    }
)

for candidate in candidate_features:
    attendance_people.append(
        {
            "id": candidate["id"],
            "name": candidate["name"],
            "feature": candidate["feature"],
            "is_admin": False
        }
    )


print()
print("==========================================")
print("ATTENDANCE PEOPLE")
print("==========================================")

for person in attendance_people:
    print(
        f"ID {person['id']} - "
        f"{person['name']}"
    )

print("==========================================")
print()


attendance_state = {}

for person in attendance_people:
    person_id = person["id"]

    attendance_state[person_id] = {
        "checked_in": False,
        "currently_visible": False,
        "visible_start": None,
        "leave_start": None,
        "check_in_time": None,
        "check_out_time": None,
        "checkout_pending": False,
        "completed_today": False
    }


today = datetime.now().strftime("%Y-%m-%d")


def create_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(
            ATTENDANCE_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "Date",
                    "ID",
                    "Name",
                    "Check In",
                    "Check Out",
                    "Duration",
                    "Status"
                ]
            )


create_attendance_file()


def find_person(feature):
    best_person = None
    best_score = -1

    for person in attendance_people:
        score = recognizer.match(
            feature,
            person["feature"],
            cv2.FaceRecognizerSF_FR_COSINE
        )

        if score > best_score:
            best_score = score
            best_person = person

    if best_person is not None:
        if best_score >= FACE_MATCH_THRESHOLD:
            return best_person, best_score

    return None, best_score


def format_duration(seconds):
    seconds = int(max(0, seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return (
        f"{hours}h "
        f"{minutes:02d}m "
        f"{seconds:02d}s"
    )


def get_person(person_id):
    for person in attendance_people:
        if person["id"] == person_id:
            return person

    return None


def check_in(person_id):
    person = get_person(person_id)

    if person is None:
        return

    state = attendance_state[person_id]

    if state["checked_in"]:
        return

    if state["completed_today"]:
        return

    now = datetime.now()

    state["checked_in"] = True
    state["currently_visible"] = True
    state["leave_start"] = None
    state["checkout_pending"] = False
    state["check_in_time"] = now
    state["check_out_time"] = None

    with open(
        ATTENDANCE_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                today,
                person["id"],
                person["name"],
                now.strftime("%H:%M:%S"),
                "",
                "",
                "Present"
            ]
        )

    print()
    print("------------------------------------------")
    print("CHECK IN SUCCESSFUL")
    print("------------------------------------------")
    print(f"ID   : {person['id']}")
    print(f"Name : {person['name']}")
    print(f"Time : {now.strftime('%H:%M:%S')}")
    print("------------------------------------------")
    print()


def update_checkout_csv(
    person_id,
    check_out_time,
    duration
):
    if not os.path.exists(ATTENDANCE_FILE):
        return

    rows = []

    with open(
        ATTENDANCE_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        for row in reader:
            try:
                row_id = int(row["ID"])
            except:
                row_id = -1

            if (
                row["Date"] == today
                and row_id == person_id
                and row["Check In"]
                and not row["Check Out"]
            ):
                row["Check Out"] = (
                    check_out_time.strftime("%H:%M:%S")
                )

                row["Duration"] = duration
                row["Status"] = "Completed"

            rows.append(row)

    with open(
        ATTENDANCE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def check_out(person_id):
    person = get_person(person_id)

    if person is None:
        return

    state = attendance_state[person_id]

    if not state["checked_in"]:
        return

    if state["check_in_time"] is None:
        return

    now = datetime.now()

    state["check_out_time"] = now

    duration_seconds = (
        now - state["check_in_time"]
    ).total_seconds()

    duration = format_duration(
        duration_seconds
    )

    update_checkout_csv(
        person_id,
        now,
        duration
    )

    state["checked_in"] = False
    state["currently_visible"] = True
    state["visible_start"] = None
    state["leave_start"] = None
    state["checkout_pending"] = False
    state["check_in_time"] = None
    state["check_out_time"] = None

    state["completed_today"] = True

    print()
    print("------------------------------------------")
    print("CHECK OUT SUCCESSFUL")
    print("------------------------------------------")
    print(f"ID       : {person['id']}")
    print(f"Name     : {person['name']}")
    print(f"Time     : {now.strftime('%H:%M:%S')}")
    print(f"Duration : {duration}")
    print("------------------------------------------")
    print()


def process_person(person, current_time):
    person_id = person["id"]

    state = attendance_state[person_id]

    state["currently_visible"] = True

    if state["completed_today"]:
        return

    if not state["checked_in"]:
        if state["visible_start"] is None:
            state["visible_start"] = current_time

            print(
                f"{person['name']} detected."
            )

            print(
                "Check-in timer started."
            )

        elapsed = (
            current_time
            - state["visible_start"]
        )

        if elapsed >= CHECK_IN_SECONDS:
            check_in(person_id)

        return

    if state["checkout_pending"]:
        if state["visible_start"] is None:
            state["visible_start"] = current_time

            print()
            print(
                f"{person['name']} returned."
            )

            print(
                "Checkout timer started."
            )

        elapsed = (
            current_time
            - state["visible_start"]
        )

        if elapsed >= CHECK_OUT_SECONDS:
            check_out(person_id)

        return

    state["leave_start"] = None


def process_missing_people(
    visible_ids,
    current_time
):
    for person in attendance_people:
        person_id = person["id"]

        state = attendance_state[person_id]

        if person_id not in visible_ids:
            state["currently_visible"] = False

            if not state["checked_in"]:
                state["visible_start"] = None
                continue

            if state["leave_start"] is None:
                state["leave_start"] = current_time

            elapsed = (
                current_time
                - state["leave_start"]
            )

            if elapsed >= LEAVE_SECONDS:
                if not state["checkout_pending"]:
                    state["checkout_pending"] = True
                    state["visible_start"] = None

                    print()
                    print(
                        f"{person['name']} "
                        f"has left the camera."
                    )

                    print(
                        "Waiting for return..."
                    )


def get_display_status(
    person_id,
    current_time
):
    state = attendance_state[person_id]

    if state["completed_today"]:
        return "COMPLETED", 0, 0

    if (
        state["checked_in"]
        and state["checkout_pending"]
        and state["visible_start"] is not None
    ):
        elapsed = (
            current_time
            - state["visible_start"]
        )

        elapsed = min(
            elapsed,
            CHECK_OUT_SECONDS
        )

        return (
            "CHECK OUT",
            elapsed,
            CHECK_OUT_SECONDS
        )

    if state["checked_in"]:
        return "CHECKED IN", 0, 0

    if state["visible_start"] is not None:
        elapsed = (
            current_time
            - state["visible_start"]
        )

        elapsed = min(
            elapsed,
            CHECK_IN_SECONDS
        )

        return (
            "CHECK IN",
            elapsed,
            CHECK_IN_SECONDS
        )

    return "DETECTED", 0, 0


def draw_person_info(
    frame,
    x,
    y,
    w,
    h,
    person,
    score,
    current_time
):
    person_id = person["id"]
    person_name = person["name"]

    status, elapsed, total = get_display_status(
        person_id,
        current_time
    )

    if status == "CHECK IN":
        box_color = (0, 255, 255)

    elif status == "CHECK OUT":
        box_color = (0, 165, 255)

    elif status == "CHECKED IN":
        box_color = (0, 255, 0)

    elif status == "COMPLETED":
        box_color = (255, 0, 255)

    else:
        box_color = (255, 255, 255)

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        box_color,
        2
    )

    name_text = (
        f"{person_name} "
        f"| ID: {person_id}"
    )

    cv2.putText(
        frame,
        name_text,
        (x, max(25, y - 35)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        box_color,
        2
    )

    status_y = max(48, y - 10)

    cv2.putText(
        frame,
        status,
        (x, status_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        box_color,
        2
    )

    if status == "CHECK IN":
        timer_text = (
            f"CHECK IN: "
            f"{elapsed:.1f} / "
            f"{total:.1f}s"
        )

        cv2.putText(
            frame,
            timer_text,
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2
        )

        progress = elapsed / total
        progress = max(0, min(1, progress))

        bar_width = w
        filled_width = int(
            bar_width * progress
        )

        bar_y = y + h + 35

        cv2.rectangle(
            frame,
            (x, bar_y),
            (x + bar_width, bar_y + 8),
            (80, 80, 80),
            -1
        )

        cv2.rectangle(
            frame,
            (x, bar_y),
            (x + filled_width, bar_y + 8),
            box_color,
            -1
        )

    elif status == "CHECK OUT":
        timer_text = (
            f"CHECK OUT: "
            f"{elapsed:.1f} / "
            f"{total:.1f}s"
        )

        cv2.putText(
            frame,
            timer_text,
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2
        )

        progress = elapsed / total
        progress = max(0, min(1, progress))

        bar_width = w
        filled_width = int(
            bar_width * progress
        )

        bar_y = y + h + 35

        cv2.rectangle(
            frame,
            (x, bar_y),
            (x + bar_width, bar_y + 8),
            (80, 80, 80),
            -1
        )

        cv2.rectangle(
            frame,
            (x, bar_y),
            (x + filled_width, bar_y + 8),
            box_color,
            -1
        )

    elif status == "CHECKED IN":
        cv2.putText(
            frame,
            "ATTENDANCE ACTIVE",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2
        )

    elif status == "COMPLETED":
        cv2.putText(
            frame,
            "ATTENDANCE COMPLETED",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2
        )

    cv2.putText(
        frame,
        f"Score: {score:.2f}",
        (x, y + h + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        box_color,
        1
    )


def start_camera():
    global camera
    global camera_running

    print("Opening webcam...")

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        camera.release()

        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return False

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    camera_running = True

    print("Webcam opened successfully.")

    return True


def stop_camera():
    global camera
    global camera_running

    camera_running = False

    if camera is not None:
        camera.release()
        camera = None

    print("Webcam stopped.")


def update_status_text():
    visible_names = []
    checked_in_names = []
    checkout_names = []
    completed_names = []

    for person in attendance_people:
        person_id = person["id"]

        state = attendance_state[person_id]

        if state["currently_visible"]:
            visible_names.append(person["name"])

        if state["checked_in"]:
            checked_in_names.append(person["name"])

        if state["checkout_pending"]:
            checkout_names.append(person["name"])

        if state["completed_today"]:
            completed_names.append(person["name"])

    if checkout_names:
        status_label.config(
            text=(
                "Checkout: "
                + ", ".join(checkout_names)
            ),
            fg="#ff9900"
        )

    elif checked_in_names:
        status_label.config(
            text=(
                "Checked in: "
                + ", ".join(checked_in_names)
            ),
            fg="#00ff88"
        )

    elif completed_names:
        status_label.config(
            text=(
                "Completed: "
                + ", ".join(completed_names)
            ),
            fg="#ff00ff"
        )

    elif visible_names:
        status_label.config(
            text=(
                "Detected: "
                + ", ".join(visible_names)
            ),
            fg="#00ffff"
        )

    else:
        status_label.config(
            text="Waiting for faces...",
            fg="white"
        )


def camera_loop():
    if not camera_running:
        return

    success, frame = camera.read()

    if not success:
        root.after(30, camera_loop)
        return

    frame = cv2.flip(frame, 1)

    original_frame = frame.copy()

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(frame)

    visible_ids = set()
    current_time = time.time()

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)

            aligned_face = recognizer.alignCrop(
                original_frame,
                face
            )

            feature = recognizer.feature(
                aligned_face
            )

            person, score = find_person(feature)

            if person is None:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "UNKNOWN",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                continue

            person_id = person["id"]

            visible_ids.add(person_id)

            process_person(
                person,
                current_time
            )

            draw_person_info(
                frame,
                x,
                y,
                w,
                h,
                person,
                score,
                current_time
            )

    process_missing_people(
        visible_ids,
        current_time
    )

    update_status_text()

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(frame_rgb)

    image = image.resize(
        (960, 720)
    )

    photo = ImageTk.PhotoImage(
        image=image
    )

    camera_label.config(
        image=photo
    )

    camera_label.image = photo

    root.after(
        30,
        camera_loop
    )


def close_program():
    stop_camera()
    root.destroy()


root = tk.Tk()

root.title("Face Attendance System")

root.geometry("1000x850")

root.configure(
    bg="#111111"
)

root.protocol(
    "WM_DELETE_WINDOW",
    close_program
)


title_label = tk.Label(
    root,
    text="FACE ATTENDANCE SYSTEM",
    font=("Arial", 24, "bold"),
    bg="#111111",
    fg="white"
)

title_label.pack(
    pady=(15, 5)
)


subtitle_label = tk.Label(
    root,
    text="Admin + Multi-candidate automatic attendance",
    font=("Arial", 12),
    bg="#111111",
    fg="#aaaaaa"
)

subtitle_label.pack(
    pady=(0, 10)
)


camera_label = tk.Label(
    root,
    bg="black",
    width=960,
    height=720
)

camera_label.pack()


status_label = tk.Label(
    root,
    text="Starting attendance system...",
    font=("Arial", 12, "bold"),
    bg="#111111",
    fg="orange"
)

status_label.pack(
    pady=10
)


print()
print("==========================================")
print("FACE ATTENDANCE SYSTEM")
print("==========================================")
print()
print(
    "Total attendance people:",
    len(attendance_people)
)
print()

if start_camera():
    status_label.config(
        text="Waiting for faces...",
        fg="white"
    )

    root.after(
        100,
        camera_loop
    )

else:
    status_label.config(
        text="Camera could not be opened.",
        fg="red"
    )


root.mainloop()
