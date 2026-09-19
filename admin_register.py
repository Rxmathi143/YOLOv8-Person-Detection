import cv2
import os
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

FACE_DETECTOR = "models/face_detection_yunet_2026may.onnx"
FACE_RECOGNIZER = "models/face_recognition_sface_2021dec.onnx"

ADMIN_IMAGE = "faces/admin.jpg"
ADMIN_FEATURE = "faces/admin.npy"


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.exists(FACE_DETECTOR):
    print("ERROR: YuNet model not found.")
    print(f"Expected: {FACE_DETECTOR}")
    exit()

if not os.path.exists(FACE_RECOGNIZER):
    print("ERROR: SFace model not found.")
    print(f"Expected: {FACE_RECOGNIZER}")
    exit()


# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs("faces", exist_ok=True)


# ============================================================
# LOAD FACE MODELS
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
# OPEN CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


print()
print("==========================================")
print("ADMIN FACE REGISTRATION")
print("==========================================")
print()
print("Look directly at the camera.")
print("Only ONE face should be visible.")
print()
print("Press SPACE to capture.")
print("Press Q to cancel.")
print("==========================================")


captured = False


# ============================================================
# CAMERA LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera.")
        break

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    # --------------------------------------------------------
    # DRAW DETECTED FACE
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

            cv2.putText(
                frame,
                "Admin Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "ADMIN REGISTRATION",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press SPACE to capture",
        (20, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to cancel",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Admin Face Registration",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # CAPTURE
    # --------------------------------------------------------

    if key == ord(" "):

        if faces is None or len(faces) == 0:

            print()
            print("ERROR: No face detected.")
            print("Please position your face in front of the camera.")
            continue


        if len(faces) > 1:

            print()
            print("ERROR: Multiple faces detected.")
            print("Only the admin should be visible.")
            continue


        # Exactly one face
        face = faces[0]


        # Align face
        aligned_face = recognizer.alignCrop(
            frame,
            face
        )


        # Extract face feature
        feature = recognizer.feature(
            aligned_face
        )


        # Save admin image
        cv2.imwrite(
            ADMIN_IMAGE,
            frame
        )


        # Save admin feature
        np.save(
            ADMIN_FEATURE,
            feature
        )


        captured = True


        print()
        print("==========================================")
        print("ADMIN REGISTRATION SUCCESSFUL")
        print("==========================================")
        print("Admin ID : ADMIN")
        print(f"Image    : {ADMIN_IMAGE}")
        print(f"Feature  : {ADMIN_FEATURE}")
        print("==========================================")
        print()


        break


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        print()
        print("Admin registration cancelled.")
        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()


if captured:

    print("Admin face is now registered.")
    print("This face will be used for admin verification.")