"""
Local face authentication (LBPH).

Commands:
  python face_auth.py capture
  python face_auth.py train
  python face_auth.py recognize
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "face_model.yml")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
CONFIDENCE_THRESHOLD = 70


def ensure_dirs() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_face_detector() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError("Could not load Haar face detector.")
    return detector


def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check that a camera is connected.")
    return cap


def detect_faces(gray: np.ndarray, detector: cv2.CascadeClassifier):
    return detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)


def largest_face(faces):
    return max(faces, key=lambda rect: rect[2] * rect[3])


def take_photo(filename: str = "photo.jpg") -> str:
    """Capture one frame from the webcam. Press SPACE to take the photo, ESC to cancel."""
    cap = open_camera()
    print("Camera open. Press SPACE to capture, ESC to cancel.")
    frame = None
    while True:
        ok, current = cap.read()
        if not ok:
            break
        preview = current.copy()
        cv2.putText(
            preview,
            "SPACE: capture  |  ESC: cancel",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Face ID camera", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            frame = current
            break
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    if frame is None:
        raise RuntimeError("Capture cancelled.")
    cv2.imwrite(filename, frame)
    return filename


def list_profiles() -> list[str]:
    if not os.path.isdir(DATASET_DIR):
        return []
    return [
        folder
        for folder in sorted(os.listdir(DATASET_DIR))
        if os.path.isdir(os.path.join(DATASET_DIR, folder))
        and folder != "__pycache__"
        and any(
            name.lower().endswith((".jpg", ".jpeg", ".png"))
            for name in os.listdir(os.path.join(DATASET_DIR, folder))
        )
    ]


def ask_profile_name() -> str:
    existing = list_profiles()
    if existing:
        print("Existing profiles:", ", ".join(existing))
    while True:
        name = input("New profile name: ").strip()
        if not name:
            print("Profile name cannot be empty.")
            continue
        invalid = set(name) & set(r'<>:"/\|?*')
        if invalid:
            print("Profile name contains invalid characters.")
            continue
        return name


def cmd_capture(user_name: str | None, count: int) -> None:
    ensure_dirs()
    if not user_name:
        user_name = ask_profile_name()
    detector = load_face_detector()
    user_dir = os.path.join(DATASET_DIR, user_name)
    os.makedirs(user_dir, exist_ok=True)

    cap = open_camera()
    saved = 0
    print(f"Capturing up to {count} face images for '{user_name}'.")
    print("Look at the camera. Press SPACE to save a face, ESC to finish.")

    while saved < count:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)
        display = frame.copy()

        if len(faces) > 0:
            x, y, w, h = largest_face(faces)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            display,
            f"Profile  {saved}/{count}  SPACE=save  ESC=exit",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Capture dataset", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        if key == 32:
            if len(faces) == 0:
                print(f"Image {saved + 1}: no face detected.")
                continue
            x, y, w, h = largest_face(faces)
            face_img = gray[y : y + h, x : x + w]
            path = os.path.join(user_dir, f"capture_{saved + 1}.jpg")
            cv2.imwrite(path, face_img)
            saved += 1
            print(f"Image {saved}: face detected and saved.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Images saved: {saved}")


def load_training_data():
    faces = []
    labels = []
    user_folders = list_profiles()
    print("Users found:", user_folders)

    for label_id, user_name in enumerate(user_folders):
        user_path = os.path.join(DATASET_DIR, user_name)
        print(f"Loading images for: {user_name}")
        for image_file in os.listdir(user_path):
            if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(user_path, image_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(label_id)

    label_map = {i: name for i, name in enumerate(user_folders)}
    return faces, labels, label_map


def cmd_train() -> None:
    ensure_dirs()
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face is missing. Install opencv-contrib-python==4.10.0.84 "
            "and uninstall opencv-python if both are present."
        )

    faces, labels, label_map = load_training_data()
    if not faces:
        raise RuntimeError("No training images were found. Run capture first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL_PATH)
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print("Training completed successfully.")
    print("Total training images:", len(faces))
    print("Model saved at:", MODEL_PATH)
    print("Label map:", label_map)


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        raise RuntimeError("Trained model not found. Run train first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    label_map = {int(k): v for k, v in raw.items()}
    return recognizer, label_map


def recognize_face(recognizer, label_map, face_image):
    prediction, confidence = recognizer.predict(face_image)
    if prediction in label_map and confidence < CONFIDENCE_THRESHOLD:
        return True, label_map[prediction], confidence
    return False, "Unknown", confidence


def cmd_recognize() -> None:
    detector = load_face_detector()
    recognizer, label_map = load_model()
    cap = open_camera()
    print("Live recognition. Press ESC to exit.")
    print("Authorized users:", label_map)

    while True:
        ok, image = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in detect_faces(gray, detector):
            face = gray[y : y + h, x : x + w]
            authorized, name, confidence = recognize_face(recognizer, label_map, face)
            color = (0, 255, 0) if authorized else (0, 0, 255)
            text = f"ACCESS GRANTED: {name}" if authorized else "ACCESS DENIED"
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                image,
                f"{text} ({confidence:.1f})",
                (x, max(24, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        cv2.imshow("Face authentication", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local LBPH face authentication")
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Create a profile and capture face images")
    capture.add_argument(
        "--user",
        default=None,
        help="Profile name. If omitted, you will be asked for a new profile name.",
    )
    capture.add_argument("--count", type=int, default=30)

    sub.add_parser("train", help="Train the LBPH model from dataset/")
    sub.add_parser("recognize", help="Live webcam authentication")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("OpenCV version:", cv2.__version__)
    print("cv2.face available:", hasattr(cv2, "face"))

    if args.command == "capture":
        cmd_capture(args.user, args.count)
    elif args.command == "train":
        cmd_train()
    elif args.command == "recognize":
        cmd_recognize()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
