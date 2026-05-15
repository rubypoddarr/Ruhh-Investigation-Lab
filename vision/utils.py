import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Prefer tiny model (fast, ~33 MB) — fall back to full YOLOv3 if available
TINY_WEIGHTS = os.path.join(MODELS_DIR, 'yolov3-tiny.weights')
TINY_CFG     = os.path.join(MODELS_DIR, 'yolov3-tiny.cfg')
FULL_WEIGHTS = os.path.join(MODELS_DIR, 'yolov3.weights')
FULL_CFG     = os.path.join(MODELS_DIR, 'yolov3.cfg')
NAMES_PATH   = os.path.join(MODELS_DIR, 'coco.names')


def load_yolo_model():
    """
    Load YOLOv3-tiny (preferred) or full YOLOv3 via OpenCV DNN.
    Returns (net, classes) or (None, []) if no weights are found.
    """
    if not os.path.exists(NAMES_PATH):
        print("[ERROR] coco.names not found.")
        return None, []

    with open(NAMES_PATH, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]

    # Try tiny model first
    if os.path.exists(TINY_WEIGHTS) and os.path.exists(TINY_CFG):
        print("[INFO] Loading YOLOv3-tiny model...")
        net = cv2.dnn.readNetFromDarknet(TINY_CFG, TINY_WEIGHTS)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[INFO] YOLOv3-tiny loaded successfully.")
        return net, classes

    # Fall back to full model
    if os.path.exists(FULL_WEIGHTS) and os.path.exists(FULL_CFG):
        print("[INFO] Loading full YOLOv3 model...")
        net = cv2.dnn.readNetFromDarknet(FULL_CFG, FULL_WEIGHTS)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[INFO] YOLOv3 loaded successfully.")
        return net, classes

    print("[WARNING] No YOLO weights found. Run download_weights.py.")
    return None, []


def get_output_layers(net):
    """Return the output layer names for YOLO."""
    layer_names = net.getLayerNames()
    try:
        return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
