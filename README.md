# Ruhh Investigation Lab

A BCA Final Year Computer Vision project using **Python Flask**, **OpenCV**, and **YOLOv3**.

Upload any image and receive:
- Object detection with green bounding boxes
- Scene classification (Indoor / Outdoor)
- Investigation-style observations

---

## Stack

| Layer      | Technology           |
|------------|----------------------|
| Backend    | Python 3, Flask      |
| CV Engine  | OpenCV DNN, YOLOv3   |
| Math       | NumPy                |
| Frontend   | HTML, CSS, JavaScript|
| Dataset    | COCO (80 classes)    |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download YOLOv3 weights

```bash
python download_weights.py
```

This downloads `yolov3.weights` (~248 MB) into the `models/` folder.
The `yolov3.cfg` and `coco.names` files are already included.

### 3. Run the app

```bash
python app.py
```

Open your browser at: `http://localhost:8000`

---

## Project Structure

```
RuhhInvestigationLab/
├── app.py                  # Flask routes
├── requirements.txt
├── download_weights.py     # One-time weight downloader
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   ├── uploads/            # Saved uploaded images
│   └── results/            # Processed output images
├── templates/
│   ├── index.html          # Landing page
│   ├── upload.html         # Upload form
│   ├── result.html         # Analysis result
│   └── about.html          # About page
├── models/
│   ├── yolov3.cfg          # YOLOv3 architecture
│   ├── yolov3.weights      # Pre-trained weights (download separately)
│   └── coco.names          # 80 COCO class labels
└── vision/
    ├── detector.py         # YOLO detection logic
    ├── analyzer.py         # Scene analysis & observations
    └── utils.py            # Model loading helpers
```

---

## Notes

- No API keys required — works fully offline after setup.
- Multiple uploads work safely — each result uses a UUID filename.
- If `yolov3.weights` is missing, the image will still be processed (without bounding boxes).
