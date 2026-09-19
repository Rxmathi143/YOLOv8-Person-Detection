import cv2
import os
import json
import time
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


# ============================================================
# PATHS
# ============================================================

FACE_DETECTOR = "models/face_detection_yunet_2026may.onnx"
FACE_RECOGNIZER = "models/face_recognition_sface_2021dec.onnx"

ADMIN_FEATURE = "faces/admin.npy"

CANDIDATE_FOLDER = "faces/candidates"
CANDIDATE_DATA = "faces/candidates/candidates.json"


# ============================================================
# SETTINGS
# ============================================================

ADMIN_THRESHOLD = 0.45
AUTO_CAPTURE_SECONDS = 2.0


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(FACE_DETECTOR):

    print("ERROR: YuNet model not found.")
    print(f"Expected: {FACE_DETECTOR}")
    exit()


if not os.path.exists(FACE_RECOGNIZER):

    print("ERROR: SFace model not found.")
    print(f"Expected: {FACE_RECOGNIZER}")
    exit()


if not os.path.exists(ADMIN_FEATURE):

    print("ERROR: Admin is not registered.")
    print()
    print("Run:")
    print("python admin_register.py")
    exit()


os.makedirs(
    CANDIDATE_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD ADMIN
# ============================================================

admin_feature = np.load(
    ADMIN_FEATURE
)


# ============================================================
# LOAD CANDIDATES
# ============================================================

if os.path.exists(CANDIDATE_DATA):

    try:

        with open(
            CANDIDATE_DATA,
            "r",
            encoding="utf-8"
        ) as file:

            candidates = json.load(file)

    except Exception:

        candidates = []

else:

    candidates = []


# ============================================================
# GET NEXT CANDIDATE ID
# ============================================================

def get_next_candidate_id():

    if not candidates:
        return 1

    ids = []

    for candidate in candidates:

        try:

            ids.append(
                int(candidate["id"])
            )

        except Exception:

            pass

    if not ids:
        return 1

    return max(ids) + 1


# ============================================================
# LOAD OPENCV MODELS
# ============================================================

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


# ============================================================
# GLOBAL VARIABLES
# ============================================================

camera = None
camera_running = False

current_frame = None

admin_verified = False

candidate_feature = None
candidate_captured = False

capture_start_time = None

verification_running = False
candidate_registration_running = False

name_window = None


# ============================================================
# FACE COMPARISON
# ============================================================

def compare_faces(feature1, feature2):

    return recognizer.match(
        feature1,
        feature2,
        cv2.FaceRecognizerSF_FR_COSINE
    )


# ============================================================
# START CAMERA
# ============================================================

def start_camera():

    global camera
    global camera_running

    if camera_running:
        return True

    print("Opening webcam...")

    # --------------------------------------------------------
    # TRY CAMERA 0 WITH DIRECTSHOW
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not camera.isOpened():

        print(
            "Camera 0 failed. Trying default camera..."
        )

        camera.release()

        camera = cv2.VideoCapture(0)

    # --------------------------------------------------------
    # CAMERA ERROR
    # --------------------------------------------------------

    if not camera.isOpened():

        print()
        print("==========================================")
        print("ERROR: CAMERA COULD NOT BE OPENED")
        print("==========================================")
        print()

        messagebox.showerror(
            "Camera Error",
            "Could not open the webcam.\n\n"
            "Make sure your camera is not being used "
            "by another application."
        )

        camera = None

        return False

    # --------------------------------------------------------
    # CAMERA RESOLUTION
    # --------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    camera_running = True

    print("Webcam opened successfully.")

    update_camera()

    return True


# ============================================================
# STOP CAMERA
# ============================================================

def stop_camera():

    global camera
    global camera_running

    camera_running = False

    if camera is not None:

        camera.release()
        camera = None

    print("Webcam stopped.")


# ============================================================
# CAMERA UPDATE
# ============================================================

def update_camera():

    global current_frame

    if not camera_running:
        return

    if camera is None:
        return

    success, frame = camera.read()

    if not success:

        print(
            "Unable to read webcam frame."
        )

        root.after(
            100,
            update_camera
        )

        return

    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )

    current_frame = frame.copy()

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(
        frame
    )

    # --------------------------------------------------------
    # DRAW FACE BOXES
    # --------------------------------------------------------

    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    # --------------------------------------------------------
    # CONVERT BGR TO RGB
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        frame_rgb
    )

    image = image.resize(
        (640, 480)
    )

    photo = ImageTk.PhotoImage(
        image=image
    )

    camera_label.config(
        image=photo
    )

    camera_label.image = photo

    # --------------------------------------------------------
    # CONTINUE CAMERA LOOP
    # --------------------------------------------------------

    root.after(
        30,
        update_camera
    )


# ============================================================
# ADMIN VERIFICATION
# ============================================================

def verify_admin():

    global admin_verified
    global verification_running

    if verification_running:
        return

    admin_verified = False
    verification_running = True

    status_label.config(
        text="Look at the camera. Verifying admin...",
        fg="orange"
    )

    register_button.config(
        state=tk.DISABLED
    )

    verify_admin_loop()


# ============================================================
# ADMIN VERIFICATION LOOP
# ============================================================

def verify_admin_loop():

    global admin_verified
    global verification_running

    if not verification_running:
        return

    if current_frame is None:

        root.after(
            100,
            verify_admin_loop
        )

        return

    frame = current_frame.copy()

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(
        frame
    )

    # --------------------------------------------------------
    # ONE FACE
    # --------------------------------------------------------

    if faces is not None and len(faces) == 1:

        face = faces[0]

        aligned_face = recognizer.alignCrop(
            frame,
            face
        )

        feature = recognizer.feature(
            aligned_face
        )

        score = compare_faces(
            admin_feature,
            feature
        )

        # ----------------------------------------------------
        # ADMIN MATCH
        # ----------------------------------------------------

        if score >= ADMIN_THRESHOLD:

            print(
                f"Admin verified. Score: {score:.3f}"
            )

            admin_verified = True
            verification_running = False

            status_label.config(
                text="Admin verified successfully!",
                fg="green"
            )

            root.after(
                1000,
                start_candidate_registration
            )

            return

        # ----------------------------------------------------
        # WRONG PERSON
        # ----------------------------------------------------

        else:

            status_label.config(
                text="Face detected, but this is not the admin.",
                fg="red"
            )

    # --------------------------------------------------------
    # MULTIPLE FACES
    # --------------------------------------------------------

    elif faces is not None and len(faces) > 1:

        status_label.config(
            text="Only the admin should be visible.",
            fg="red"
        )

    # --------------------------------------------------------
    # NO FACE
    # --------------------------------------------------------

    else:

        status_label.config(
            text="Please show the admin face.",
            fg="orange"
        )

    root.after(
        100,
        verify_admin_loop
    )


# ============================================================
# START CANDIDATE REGISTRATION
# ============================================================

def start_candidate_registration():

    global candidate_captured
    global candidate_feature
    global capture_start_time
    global candidate_registration_running

    candidate_captured = False
    candidate_feature = None
    capture_start_time = None

    candidate_registration_running = True

    status_label.config(
        text="Admin verified. Candidate, stand in front of camera.",
        fg="green"
    )

    candidate_capture_loop()


# ============================================================
# CANDIDATE CAPTURE
# ============================================================

def candidate_capture_loop():

    global capture_start_time
    global candidate_feature
    global candidate_captured
    global candidate_registration_running

    if not candidate_registration_running:
        return

    if candidate_captured:
        return

    if current_frame is None:

        root.after(
            100,
            candidate_capture_loop
        )

        return

    frame = current_frame.copy()

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(
        frame
    )

    # ========================================================
    # NO FACE
    # ========================================================

    if faces is None or len(faces) == 0:

        capture_start_time = None

        status_label.config(
            text="Waiting for candidate face...",
            fg="orange"
        )

        root.after(
            100,
            candidate_capture_loop
        )

        return

    # ========================================================
    # MULTIPLE FACES
    # ========================================================

    if len(faces) > 1:

        capture_start_time = None

        status_label.config(
            text="Only one candidate should be visible.",
            fg="red"
        )

        root.after(
            100,
            candidate_capture_loop
        )

        return

    # ========================================================
    # ONE FACE
    # ========================================================

    face = faces[0]

    aligned_face = recognizer.alignCrop(
        frame,
        face
    )

    feature = recognizer.feature(
        aligned_face
    )

    # ========================================================
    # CHECK IF ADMIN
    # ========================================================

    admin_score = compare_faces(
        admin_feature,
        feature
    )

    if admin_score >= ADMIN_THRESHOLD:

        capture_start_time = None

        status_label.config(
            text=(
                "Admin detected. Please stand aside "
                "and let the candidate enter."
            ),
            fg="red"
        )

        root.after(
            100,
            candidate_capture_loop
        )

        return

    # ========================================================
    # START AUTO CAPTURE TIMER
    # ========================================================

    if capture_start_time is None:

        capture_start_time = time.time()

    elapsed = (
        time.time() - capture_start_time
    )

    remaining = (
        AUTO_CAPTURE_SECONDS - elapsed
    )

    # ========================================================
    # WAIT FOR 2 SECONDS
    # ========================================================

    if remaining > 0:

        status_label.config(
            text=(
                f"Candidate detected. "
                f"Capturing in {remaining:.1f}s..."
            ),
            fg="orange"
        )

        root.after(
            50,
            candidate_capture_loop
        )

        return

    # ========================================================
    # CAPTURE COMPLETE
    # ========================================================

    candidate_feature = feature

    candidate_captured = True
    candidate_registration_running = False

    status_label.config(
        text="Candidate face captured successfully!",
        fg="green"
    )

    print(
        "Candidate face captured."
    )

    # ========================================================
    # OPEN NAME POPUP
    # ========================================================

    show_name_keypad()


# ============================================================
# NAME + ON-SCREEN KEYBOARD POPUP
# ============================================================

def show_name_keypad():

    global name_window

    # --------------------------------------------------------
    # PREVENT DUPLICATE POPUP
    # --------------------------------------------------------

    if name_window is not None:

        try:

            if name_window.winfo_exists():
                name_window.focus_force()
                return

        except Exception:

            pass

    # --------------------------------------------------------
    # CREATE POPUP
    # --------------------------------------------------------

    name_window = tk.Toplevel(
        root
    )

    name_window.title(
        "Enter Candidate Name"
    )

    name_window.geometry(
        "650x560"
    )

    name_window.configure(
        bg="#111111"
    )

    name_window.resizable(
        False,
        False
    )

    # --------------------------------------------------------
    # CENTER POPUP
    # --------------------------------------------------------

    root.update_idletasks()

    root_x = root.winfo_x()
    root_y = root.winfo_y()

    root_width = root.winfo_width()
    root_height = root.winfo_height()

    popup_width = 650
    popup_height = 560

    center_x = (
        root_x
        + (root_width - popup_width) // 2
    )

    center_y = (
        root_y
        + (root_height - popup_height) // 2
    )

    name_window.geometry(
        f"{popup_width}x{popup_height}"
        f"+{center_x}+{center_y}"
    )

    # --------------------------------------------------------
    # MAKE POPUP MODAL
    # --------------------------------------------------------

    name_window.transient(
        root
    )

    name_window.grab_set()

    name_window.focus_force()

    # ========================================================
    # TITLE
    # ========================================================

    title = tk.Label(
        name_window,
        text="ENTER CANDIDATE NAME",
        font=("Arial", 20, "bold"),
        bg="#111111",
        fg="white"
    )

    title.pack(
        pady=(15, 8)
    )

    # ========================================================
    # INFORMATION
    # ========================================================

    info = tk.Label(
        name_window,
        text="Use the keyboard below to enter the name",
        font=("Arial", 10),
        bg="#111111",
        fg="#aaaaaa"
    )

    info.pack(
        pady=(0, 5)
    )

    # ========================================================
    # NAME ENTRY
    # ========================================================

    name_entry = tk.Entry(
        name_window,
        font=("Arial", 20),
        width=28,
        justify="center"
    )

    name_entry.pack(
        pady=8
    )

    name_entry.focus_set()

    # ========================================================
    # KEYPAD FRAME
    # ========================================================

    keypad = tk.Frame(
        name_window,
        bg="#111111"
    )

    keypad.pack(
        pady=5
    )

    # ========================================================
    # ADD CHARACTER
    # ========================================================

    def add_character(character):

        name_entry.insert(
            tk.END,
            character
        )

        name_entry.focus_set()

    # ========================================================
    # BACKSPACE
    # ========================================================

    def backspace():

        current = name_entry.get()

        if current:

            name_entry.delete(
                len(current) - 1,
                tk.END
            )

        name_entry.focus_set()

    # ========================================================
    # CLEAR
    # ========================================================

    def clear_name():

        name_entry.delete(
            0,
            tk.END
        )

        name_entry.focus_set()

    # ========================================================
    # KEYS
    # ========================================================

    keys = [

        [
            "A", "B", "C", "D",
            "E", "F", "G", "H", "I"
        ],

        [
            "J", "K", "L", "M",
            "N", "O", "P", "Q", "R"
        ],

        [
            "S", "T", "U", "V",
            "W", "X", "Y", "Z", "0"
        ],

        [
            "1", "2", "3", "4",
            "5", "6", "7", "8", "9"
        ]

    ]

    # ========================================================
    # CREATE KEYS
    # ========================================================

    for row_index, row in enumerate(keys):

        for column_index, key in enumerate(row):

            button = tk.Button(
                keypad,
                text=key,
                font=("Arial", 11, "bold"),
                width=4,
                height=1,
                command=lambda value=key:
                    add_character(value)
            )

            button.grid(
                row=row_index,
                column=column_index,
                padx=2,
                pady=2
            )

    # ========================================================
    # SPACE
    # ========================================================

    space_button = tk.Button(
        keypad,
        text="SPACE",
        font=("Arial", 10, "bold"),
        width=11,
        height=1,
        command=lambda:
            add_character(" ")
    )

    space_button.grid(
        row=4,
        column=0,
        columnspan=3,
        padx=2,
        pady=5
    )

    # ========================================================
    # BACKSPACE
    # ========================================================

    backspace_button = tk.Button(
        keypad,
        text="BACKSPACE",
        font=("Arial", 10, "bold"),
        width=11,
        height=1,
        command=backspace
    )

    backspace_button.grid(
        row=4,
        column=3,
        columnspan=3,
        padx=2,
        pady=5
    )

    # ========================================================
    # CLEAR
    # ========================================================

    clear_button = tk.Button(
        keypad,
        text="CLEAR",
        font=("Arial", 10, "bold"),
        width=11,
        height=1,
        command=clear_name
    )

    clear_button.grid(
        row=4,
        column=6,
        columnspan=3,
        padx=2,
        pady=5
    )

    # ========================================================
    # SAVE CANDIDATE
    # ========================================================

    def save_from_popup():

        global candidate_feature
        global name_window

        # ----------------------------------------------------
        # CHECK FACE
        # ----------------------------------------------------

        if candidate_feature is None:

            messagebox.showerror(
                "Error",
                "Candidate face was not captured.",
                parent=name_window
            )

            return

        # ----------------------------------------------------
        # GET NAME
        # ----------------------------------------------------

        name = name_entry.get().strip()

        if not name:

            messagebox.showwarning(
                "Name Required",
                "Please enter the candidate name.",
                parent=name_window
            )

            return

        # ----------------------------------------------------
        # CREATE ID
        # ----------------------------------------------------

        candidate_id = get_next_candidate_id()

        # ----------------------------------------------------
        # FEATURE FILE
        # ----------------------------------------------------

        feature_filename = (
            f"candidate_{candidate_id}.npy"
        )

        feature_path = os.path.join(
            CANDIDATE_FOLDER,
            feature_filename
        )

        # ----------------------------------------------------
        # SAVE FACE FEATURE
        # ----------------------------------------------------

        np.save(
            feature_path,
            candidate_feature
        )

        # ----------------------------------------------------
        # CREATE CANDIDATE DATA
        # ----------------------------------------------------

        candidate_data = {

            "id": candidate_id,

            "name": name,

            "feature": feature_filename

        }

        candidates.append(
            candidate_data
        )

        # ----------------------------------------------------
        # SAVE JSON
        # ----------------------------------------------------

        with open(
            CANDIDATE_DATA,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                candidates,
                file,
                indent=4
            )

        print()
        print("==========================================")
        print("CANDIDATE REGISTRATION SUCCESSFUL")
        print("==========================================")
        print(f"Candidate ID   : {candidate_id}")
        print(f"Candidate Name : {name}")
        print(f"Feature File   : {feature_filename}")
        print("==========================================")
        print()

        # ----------------------------------------------------
        # SUCCESS MESSAGE
        # ----------------------------------------------------

        messagebox.showinfo(
            "Registration Successful",
            f"Candidate registered successfully!\n\n"
            f"ID   : {candidate_id}\n"
            f"Name : {name}",
            parent=name_window
        )

        # ----------------------------------------------------
        # CLOSE POPUP
        # ----------------------------------------------------

        try:

            name_window.grab_release()

        except Exception:

            pass

        name_window.destroy()

        name_window = None

        # ----------------------------------------------------
        # RESET REGISTRATION
        # ----------------------------------------------------

        reset_registration()

    # ========================================================
    # SAVE BUTTON
    # ========================================================

    save_button = tk.Button(
        name_window,
        text="SAVE CANDIDATE",
        font=("Arial", 13, "bold"),
        bg="#16a34a",
        fg="white",
        activebackground="#15803d",
        activeforeground="white",
        width=22,
        height=2,
        command=save_from_popup
    )

    save_button.pack(
        pady=(5, 3)
    )

    # ========================================================
    # CANCEL POPUP
    # ========================================================

    def cancel_popup():

        global name_window
        global candidate_feature
        global candidate_captured

        try:

            name_window.grab_release()

        except Exception:

            pass

        name_window.destroy()

        name_window = None

        # ----------------------------------------------------
        # RESET CANDIDATE REGISTRATION
        # ----------------------------------------------------

        candidate_feature = None
        candidate_captured = False

        reset_registration()

    # ========================================================
    # CANCEL BUTTON
    # ========================================================

    cancel_button = tk.Button(
        name_window,
        text="CANCEL",
        font=("Arial", 11),
        width=12,
        height=1,
        command=cancel_popup
    )

    cancel_button.pack(
        pady=3
    )

    # ========================================================
    # CLOSE X BUTTON
    # ========================================================

    name_window.protocol(
        "WM_DELETE_WINDOW",
        cancel_popup
    )


# ============================================================
# RESET REGISTRATION
# ============================================================

def reset_registration():

    global candidate_feature
    global candidate_captured
    global capture_start_time
    global admin_verified
    global verification_running
    global candidate_registration_running

    candidate_feature = None

    candidate_captured = False

    capture_start_time = None

    admin_verified = False

    verification_running = False

    candidate_registration_running = False

    register_button.config(
        state=tk.NORMAL
    )

    status_label.config(
        text="Ready. Click REGISTER CANDIDATE.",
        fg="white"
    )


# ============================================================
# CLOSE PROGRAM
# ============================================================

def close_program():

    global name_window

    # --------------------------------------------------------
    # CLOSE NAME POPUP IF OPEN
    # --------------------------------------------------------

    if name_window is not None:

        try:

            name_window.grab_release()

        except Exception:

            pass

        try:

            name_window.destroy()

        except Exception:

            pass

        name_window = None

    # --------------------------------------------------------
    # STOP CAMERA
    # --------------------------------------------------------

    stop_camera()

    # --------------------------------------------------------
    # CLOSE MAIN WINDOW
    # --------------------------------------------------------

    root.destroy()


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Candidate Registration"
)

root.geometry(
    "900x900"
)

root.configure(
    bg="#111111"
)

root.protocol(
    "WM_DELETE_WINDOW",
    close_program
)


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    root,
    text="CANDIDATE REGISTRATION",
    font=("Arial", 24, "bold"),
    bg="#111111",
    fg="white"
)

title_label.pack(
    pady=(15, 5)
)


# ============================================================
# SUBTITLE
# ============================================================

subtitle_label = tk.Label(
    root,
    text="Admin controlled face registration",
    font=("Arial", 12),
    bg="#111111",
    fg="#aaaaaa"
)

subtitle_label.pack(
    pady=(0, 10)
)


# ============================================================
# CAMERA
# ============================================================

camera_label = tk.Label(
    root,
    bg="black",
    width=640,
    height=480
)

camera_label.pack()


# ============================================================
# REGISTER BUTTON
# ============================================================

register_button = tk.Button(
    root,
    text="REGISTER CANDIDATE",
    font=("Arial", 14, "bold"),
    bg="#2563eb",
    fg="white",
    activebackground="#1d4ed8",
    activeforeground="white",
    width=25,
    height=2,
    command=verify_admin
)

register_button.pack(
    pady=(12, 3)
)


# ============================================================
# MESSAGE
# ============================================================

message_label = tk.Label(
    root,
    text="Register your candidates faces",
    font=("Arial", 10),
    bg="#111111",
    fg="#aaaaaa"
)

message_label.pack()


# ============================================================
# STATUS
# ============================================================

status_label = tk.Label(
    root,
    text="Camera starting...",
    font=("Arial", 11),
    bg="#111111",
    fg="orange"
)

status_label.pack(
    pady=8
)


# ============================================================
# START GUI
# ============================================================

print()
print("==========================================")
print("CANDIDATE REGISTRATION SYSTEM")
print("==========================================")
print("Starting camera...")
print()

root.after(
    500,
    start_camera
)

root.mainloop()