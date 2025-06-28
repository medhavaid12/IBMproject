
import sqlite3
from flask import Flask, request, jsonify, render_template, Response
from datetime import datetime
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import mediapipe as mp

# CONFIG
DB_FILE = "mask_violations.db"
MODEL_PATH = "C:/Users/BONNYJOY/Documents/mask_violation_project/runs/detect/mask_detector/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
INFERENCE_IMG_SIZE = 640

app = Flask(__name__, template_folder='templates')
model = YOLO(MODEL_PATH)
class_names = model.names
camera = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def log_violation(label):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT,
                timestamp TEXT
            )
        """)
        cursor.execute("INSERT INTO violations (label, timestamp) VALUES (?, ?)", (label, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ DB error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image received'}), 400

        base64_str = data['image'].split(',')[1]
        decoded = base64.b64decode(base64_str)
        np_arr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Step 1: Check for face presence using MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detector.process(frame_rgb)

        if not face_results.detections:
            return jsonify({'detections': [], 'no_face': True})  # ✅ Exit early if no face

        # Step 2: Convert MediaPipe faces to box coordinates
        face_boxes = []
        ih, iw, _ = frame.shape
        for det in face_results.detections:
            bbox = det.location_data.relative_bounding_box
            fx = int(bbox.xmin * iw)
            fy = int(bbox.ymin * ih)
            fw = int(bbox.width * iw)
            fh = int(bbox.height * ih)
            face_boxes.append((fx, fy, fx + fw, fy + fh))

        # Step 3: YOLO inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, imgsz=INFERENCE_IMG_SIZE, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = class_names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Step 4: Only keep boxes that overlap with face
            matched = False
            for fx1, fy1, fx2, fy2 in face_boxes:
                if x1 < fx2 and x2 > fx1 and y1 < fy2 and y2 > fy1:
                    matched = True
                    break

            if matched:
                detections.append({
                    'x': x1, 'y': y1,
                    'width': x2 - x1, 'height': y2 - y1,
                    'label': label, 'confidence': round(conf * 100, 2)
                })
                if label == "no_mask":
                    log_violation(label)

        # If face was detected, but YOLO missed it
        return jsonify({
            'detections': detections,
            'no_face': False
        })

    except Exception as e:
        print(f"❌ Detection error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("✅ App running at http://127.0.0.1:4040")
    app.run(host='127.0.0.1', port=4040, debug=True)
    if camera.isOpened():
        camera.release()
    cv2.destroyAllWindows()

import sqlite3
from flask import Flask, request, jsonify, render_template, Response
from datetime import datetime
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import mediapipe as mp

# CONFIG
DB_FILE = "mask_violations.db"
MODEL_PATH = "C:/Users/BONNYJOY/Documents/mask_violation_project/runs/detect/mask_detector/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
INFERENCE_IMG_SIZE = 640

app = Flask(__name__, template_folder='templates')
model = YOLO(MODEL_PATH)
class_names = model.names
camera = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def log_violation(label):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT,
                timestamp TEXT
            )
        """)
        cursor.execute("INSERT INTO violations (label, timestamp) VALUES (?, ?)", (label, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ DB error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image received'}), 400

        base64_str = data['image'].split(',')[1]
        decoded = base64.b64decode(base64_str)
        np_arr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Step 1: Check for face presence using MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detector.process(frame_rgb)

        if not face_results.detections:
            return jsonify({'detections': [], 'no_face': True})  # ✅ Exit early if no face

        # Step 2: Convert MediaPipe faces to box coordinates
        face_boxes = []
        ih, iw, _ = frame.shape
        for det in face_results.detections:
            bbox = det.location_data.relative_bounding_box
            fx = int(bbox.xmin * iw)
            fy = int(bbox.ymin * ih)
            fw = int(bbox.width * iw)
            fh = int(bbox.height * ih)
            face_boxes.append((fx, fy, fx + fw, fy + fh))

        # Step 3: YOLO inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, imgsz=INFERENCE_IMG_SIZE, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = class_names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Step 4: Only keep boxes that overlap with face
            matched = False
            for fx1, fy1, fx2, fy2 in face_boxes:
                if x1 < fx2 and x2 > fx1 and y1 < fy2 and y2 > fy1:
                    matched = True
                    break

            if matched:
                detections.append({
                    'x': x1, 'y': y1,
                    'width': x2 - x1, 'height': y2 - y1,
                    'label': label, 'confidence': round(conf * 100, 2)
                })
                if label == "no_mask":
                    log_violation(label)

        # If face was detected, but YOLO missed it
        return jsonify({
            'detections': detections,
            'no_face': False
        })

    except Exception as e:
        print(f"❌ Detection error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("✅ App running at http://127.0.0.1:4040")
    app.run(host='127.0.0.1', port=4040, debug=True)
    if camera.isOpened():
        camera.release()
    cv2.destroyAllWindows()

