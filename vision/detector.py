import cv2
import numpy as np
import os
import uuid
from vision.utils import load_yolo_model, get_output_layers

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

# Colour palette for bounding boxes (cycles through classes)
COLORS = [
    (0, 255, 0),    # green
    (0, 200, 255),  # cyan
    (255, 100, 0),  # orange
    (180, 0, 255),  # purple
    (255, 255, 0),  # yellow
]


def detect_objects(image_path, result_folder):
    """
    Run YOLO object detection on the given image.
    Draws labelled bounding boxes and saves the annotated result.
    Returns (result_filename, list_of_detected_labels).
    """
    image = cv2.imread(image_path)
    if image is None:
        return '', []

    height, width = image.shape[:2]
    net, classes = load_yolo_model()

    if net is None:
        # Save original image as result with a watermark
        cv2.putText(image, 'YOLO weights not loaded', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return _save_result(image, result_folder), []

    # Build input blob for YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                  swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(get_output_layers(net))

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence > CONFIDENCE_THRESHOLD:
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w  = int(detection[2] * width)
                h  = int(detection[3] * height)
                x  = int(cx - w / 2)
                y  = int(cy - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    detected_labels = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            conf  = confidences[i]
            detected_labels.append(label)

            color = COLORS[class_ids[i] % len(COLORS)]

            # Bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Label pill background
            text = f"{label}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            y_label = max(y, th + 8)
            cv2.rectangle(image,
                          (x, y_label - th - 6), (x + tw + 8, y_label + 2),
                          color, -1)
            cv2.putText(image, text, (x + 4, y_label - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1,
                        cv2.LINE_AA)

    return _save_result(image, result_folder), detected_labels


def _save_result(image, result_folder):
    filename = f"result_{uuid.uuid4().hex}.jpg"
    path = os.path.join(result_folder, filename)
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return filename
