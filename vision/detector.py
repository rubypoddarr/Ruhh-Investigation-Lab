import cv2
import os
import uuid
import threading
from ultralytics import YOLO

# -------------------------------
# CONFIG
# -------------------------------

CONFIDENCE_THRESHOLD = 0.35

COLORS = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 100, 0),
    (180, 0, 255),
    (255, 255, 0),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "yolov8n.pt")

# -------------------------------
# GLOBAL MODEL STATE (SAFE LAZY LOADING)
# -------------------------------

model = None
model_lock = threading.Lock()

def get_model():
    """
    Lazy-load YOLO model only once.
    Safe for Gunicorn multi-thread environment.
    """
    global model

    with model_lock:
        if model is None:
            model = YOLO(MODEL_PATH)
            model.to("cpu")
    return model


# -------------------------------
# MAIN DETECTION FUNCTION
# -------------------------------

def detect_objects(image_path, result_folder):

    os.makedirs(result_folder, exist_ok=True)

    image = cv2.imread(image_path)

    if image is None:
        return "", []

    # -------------------------------
    # SAFE RESIZE
    # -------------------------------

    h, w = image.shape[:2]
    max_size = 640

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # -------------------------------
    # LOAD MODEL SAFELY
    # -------------------------------

    yolo = get_model()

    # -------------------------------
    # INFERENCE (CPU SAFE)
    # -------------------------------

    results = yolo.predict(
        source=image,
        imgsz=320,
        conf=CONFIDENCE_THRESHOLD,
        device="cpu",
        verbose=False
    )

    detected_labels = []

    # -------------------------------
    # DRAW RESULTS
    # -------------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            label = yolo.names[class_id]

            detected_labels.append(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = COLORS[class_id % len(COLORS)]

            # draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {confidence:.0%}"

            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            cv2.rectangle(
                image,
                (x1, y1 - th - 10),
                (x1 + tw + 10, y1),
                color,
                -1
            )

            cv2.putText(
                image,
                text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

    return _save_result(image, result_folder), detected_labels


# -------------------------------
# SAVE RESULT IMAGE
# -------------------------------

def _save_result(image, result_folder):

    filename = f"result_{uuid.uuid4().hex}.jpg"
    path = os.path.join(result_folder, filename)

    cv2.imwrite(
        path,
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )

    return filename
