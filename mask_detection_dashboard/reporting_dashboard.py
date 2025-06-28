import sqlite3
from flask import Flask, request, jsonify, render_template
from datetime import datetime
import cv2
import numpy as np
import base64
from ultralytics import YOLO

# ======================= CONFIG =======================
DB_FILE = "mask_violations.db"
MODEL_PATH = r"C:\\Users\\BONNYJOY\\Documents\\mask_violation_project\\runs\\detect\\train4\\weights\\best.pt"

# ======================= Initialize Flask App and Model =======================

app = Flask(__name__, template_folder='templates')
model = YOLO(MODEL_PATH)
print("✅ YOLO model loaded successfully.")

class_names = ['mask', 'no_mask']

# ======================= DB Logger =======================
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
        print(f"\u274c DB Log Error: {e}")

# ======================= HTML Frontend =======================
@app.route("/")
def index():
    return render_template("index.html")

# ======================= Detection Endpoint =======================
@app.route("/detect", methods=["POST"])
def detect_from_webcam_frame():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400

        image_data = data['image'].split(',')[1]
        decoded = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(frame)[0]
        detections = []

        if results and results.boxes:
            for box in results.boxes:
                if box.conf is None or box.cls is None:
                    continue
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_id < 0 or cls_id >= len(class_names):
                    continue

                label = class_names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                width = x2 - x1
                height = y2 - y1

                if label == "no_mask":
                    log_violation(label)

                print(f"\U0001F50D Detected: {label} ({conf:.2f}) at ({x1},{y1},{x2},{y2})")

                detections.append({
                    'x': x1,
                    'y': y1,
                    'width': width,
                    'height': height,
                    'label': label,
                    'confidence': round(conf * 100, 2)
                })
        else:
            print("⚠️ No boxes detected!")

        return jsonify({'detections': detections})
    except Exception as e:
        print(f"❌ Exception: {e}")
        return jsonify({'error': str(e)}), 500

# ======================= Run App =======================
if __name__ == "__main__":
    print("\u2705 Starting Face Mask Detection Dashboard at http://127.0.0.1:4040")
    app.run(host="127.0.0.1", port=4040, debug=True)
