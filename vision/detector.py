import cv2
import os
import uuid
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.35

COLORS = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 100, 0),
    (180, 0, 255),
    (255, 255, 0),
]

# -------------------------------
# BASE DIRECTORY
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "yolov8n.pt")

# -------------------------------
# LOAD MODEL ONLY ONCE
# -------------------------------

model = YOLO(MODEL_PATH)

# FORCE CPU MODE FOR RENDER
model.to("cpu")


def detect_objects(image_path, result_folder):

    # Ensure result folder exists
    os.makedirs(result_folder, exist_ok=True)

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        return '', []

    # ---------------------------------
    # RESIZE IMAGE
    # Helps avoid Render RAM crashes
    # ---------------------------------

    height, width = image.shape[:2]

    max_size = 640

    if width > max_size or height > max_size:

        scale = min(max_size / width, max_size / height)

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height)
        )

        # overwrite optimized image
        cv2.imwrite(image_path, image)

    # ---------------------------------
    # RUN YOLO DETECTION
    # ---------------------------------

    results = model.predict(
        source=image,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
        device="cpu"
    )

    detected_labels = []

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])

            label = model.names[class_id]

            detected_labels.append(label)

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            color = COLORS[class_id % len(COLORS)]

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Label text
            text = f"{label} {confidence:.0%}"

            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            # Label background
            cv2.rectangle(
                image,
                (x1, y1 - th - 10),
                (x1 + tw + 10, y1),
                color,
                -1
            )

            # Label text draw
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


def _save_result(image, result_folder):

    filename = f"result_{uuid.uuid4().hex}.jpg"

    path = os.path.join(
        result_folder,
        filename
    )

    cv2.imwrite(
        path,
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )

    return filename