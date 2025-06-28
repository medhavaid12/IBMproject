<<<<<<< HEAD
# web_mask_detector.py

import cv2
import numpy as np 
from flask import Flask, render_template, Response, jsonify # Added jsonify for API endpoint
from flask_socketio import SocketIO
import base64
from threading import Lock, Thread
import eventlet # Important for performance
import sqlite3 # For database logging
from datetime import datetime # For timestamps in database
from collections import Counter # For statistics analysis (if needed on backend, but mostly for frontend display)

# We need to monkey patch the server for eventlet to work with threading
eventlet.monkey_patch()

# --- Configuration ---
DB_FILE = "mask_violations.db" # SQLite database file for logging
FACE_DETECTOR_PROTOTXT = "deploy.prototxt.txt" # Caffe model prototxt
FACE_DETECTOR_MODEL = "res10_300x300_ssd_iter_140000.caffemodel" # Caffe model weights
MASK_CLASSIFIER_MODEL = "face_mask_detector.h5" # Keras mask classifier model

# --- Initialization ---
app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app)

# Threading lock to ensure thread-safe exchanges of frames
thread_lock = Lock()
video_thread = None # Stores the background thread for video processing

# Load the models once at the start
print("[INFO] Loading models...")
try:
    face_detector_net = cv2.dnn.readNet(FACE_DETECTOR_PROTOTXT, FACE_DETECTOR_MODEL)
    print("[INFO] Models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")
    # Exit or handle gracefully if models can't be loaded
    exit(1)

# --- SQLite Database Logging Function ---
def log_violation(label):
    """
    Logs a mask violation to the SQLite database.
    Creates the table if it doesn't exist.
    """
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
        print(f"❌ DB Log Error: {e}")

# --- Video Processing Thread ---
def video_processing_thread():
    """
    This function runs in a background thread to continuously process frames
    from the webcam, perform detection, and emit results via SocketIO.
    """
    video_capture = cv2.VideoCapture(0) # 0 for default webcam
    if not video_capture.isOpened():
        print("[ERROR] Cannot open webcam.")
        # Emit an error event to the frontend
        socketio.emit('error_message', {'message': 'Cannot open webcam. Please check connection and permissions.'})
        return

    print("[INFO] Video thread started. Streaming to dashboard...")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[INFO] Failed to read frame from video capture. Breaking stream.")
            break

        (h, w) = frame.shape[:2]
        # Preprocessing for Caffe model
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_detector_net.setInput(blob)
        detections = face_detector_net.forward()

        faces = []
        locations = []
        predictions = []

        # Process all detected faces first
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # Filter out weak detections
            if confidence > 0.5: # Face detection confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                if face.size == 0: # Skip empty faces
                    continue

                # Preprocessing for Keras mask classifier
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = np.expand_dims(face, axis=0) # Add batch dimension
                face = face / 255.0 # Normalize

                faces.append(face)
                locations.append((startX, startY, endX, endY))
        

        # Initialize counters for the dashboard
        mask_count = 0
        no_mask_count = 0

        # Loop over the locations and their corresponding predictions to draw and count
        for (box, pred) in zip(locations, predictions):
            (startX, startY, endX, endY) = box
            mask_prob = pred[0] # The model outputs a single value due to sigmoid

            if mask_prob > 0.5: # Mask classification confidence threshold
                label = "Mask"
                color = (0, 255, 0) # Green for Mask
                mask_count += 1
            else:
                label = "No Mask"
                color = (0, 0, 255) # Red for No Mask
                no_mask_count += 1
                # Log violation to database
                log_violation(label)
                # Emit event for real-time log display on dashboard
                socketio.emit('log_event', {'message': f'Violation: No Mask Detected at {datetime.now().strftime("%H:%M:%S")}!'})

            # Format label for display on frame
            label_text = f"{label}: {mask_prob * 100:.2f}%" # Show confidence as 0.00%
            cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # --- Emit Data to Frontend ---
        # 1. Update general statistics
        socketio.emit('update_stats', {
            'total_faces': len(faces),
            'mask_count': mask_count,
            'no_mask_count': no_mask_count
        })

        # 2. Stream the video frame (base64 encoded)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'image': frame_b64})

        # Yield control to the eventlet server to prevent blocking
        socketio.sleep(0.01) # Small sleep to allow other events to be processed

    video_capture.release()
    print("[INFO] Video processing thread stopped.")

# --- Flask Routes and SocketIO Events ---
@app.route('/')
def index():
    """Serves the main dashboard page."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handles a new client connection. Starts the background video processing thread if not already running."""
    global video_thread
    with thread_lock:
        if video_thread is None:
            video_thread = socketio.start_background_task(target=video_processing_thread)
    print("Client connected.")

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnection."""
    print("Client disconnected.")

# --- API Endpoint for Historical Violation Statistics ---
@app.route('/api/violations_history')
def api_violations_history():
    """
    API endpoint to fetch all historical mask violation records from the database.
    Returns data as JSON for frontend display.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT label, timestamp FROM violations ORDER BY timestamp DESC")
        records = cursor.fetchall()
        conn.close()

        # Format data for JSON response
        formatted_records = []
        for label, timestamp_str in records:
            dt_object = datetime.fromisoformat(timestamp_str)
            formatted_records.append({
                'label': label,
                'timestamp': dt_object.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Optional: Add summary counts to the history endpoint as well
        total_violations_count = len(records)
        no_mask_violations_count = sum(1 for r in records if r[0] == 'No Mask')
        
        # Count by date for daily summary
        date_counts = Counter(datetime.fromisoformat(r[1]).strftime('%Y-%m-%d') for r in records)
        sorted_date_counts = dict(sorted(date_counts.items()))

        return jsonify({
            'history': formatted_records,
            'summary': {
                'total_logged_incidents': total_violations_count,
                'total_no_mask_violations': no_mask_violations_count,
                'violations_by_date': sorted_date_counts
            }
        })

    except Exception as e:
        print(f"❌ Error fetching historical violations: {e}")
        return jsonify({'error': str(e)}), 500


# --- Main Execution ---
if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    # Use eventlet as the web server for performance with SocketIO
    # Host '0.0.0.0' makes it accessible on your network (useful for testing on other devices)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
=======
# web_mask_detector.py

import cv2
import numpy as np 
from flask import Flask, render_template, Response, jsonify # Added jsonify for API endpoint
from flask_socketio import SocketIO
import base64
from threading import Lock, Thread
import eventlet # Important for performance
import sqlite3 # For database logging
from datetime import datetime # For timestamps in database
from collections import Counter # For statistics analysis (if needed on backend, but mostly for frontend display)

# We need to monkey patch the server for eventlet to work with threading
eventlet.monkey_patch()

# --- Configuration ---
DB_FILE = "mask_violations.db" # SQLite database file for logging
FACE_DETECTOR_PROTOTXT = "deploy.prototxt.txt" # Caffe model prototxt
FACE_DETECTOR_MODEL = "res10_300x300_ssd_iter_140000.caffemodel" # Caffe model weights
MASK_CLASSIFIER_MODEL = "face_mask_detector.h5" # Keras mask classifier model

# --- Initialization ---
app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app)

# Threading lock to ensure thread-safe exchanges of frames
thread_lock = Lock()
video_thread = None # Stores the background thread for video processing

# Load the models once at the start
print("[INFO] Loading models...")
try:
    face_detector_net = cv2.dnn.readNet(FACE_DETECTOR_PROTOTXT, FACE_DETECTOR_MODEL)
    print("[INFO] Models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")
    # Exit or handle gracefully if models can't be loaded
    exit(1)

# --- SQLite Database Logging Function ---
def log_violation(label):
    """
    Logs a mask violation to the SQLite database.
    Creates the table if it doesn't exist.
    """
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
        print(f"❌ DB Log Error: {e}")

# --- Video Processing Thread ---
def video_processing_thread():
    """
    This function runs in a background thread to continuously process frames
    from the webcam, perform detection, and emit results via SocketIO.
    """
    video_capture = cv2.VideoCapture(0) # 0 for default webcam
    if not video_capture.isOpened():
        print("[ERROR] Cannot open webcam.")
        # Emit an error event to the frontend
        socketio.emit('error_message', {'message': 'Cannot open webcam. Please check connection and permissions.'})
        return

    print("[INFO] Video thread started. Streaming to dashboard...")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[INFO] Failed to read frame from video capture. Breaking stream.")
            break

        (h, w) = frame.shape[:2]
        # Preprocessing for Caffe model
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_detector_net.setInput(blob)
        detections = face_detector_net.forward()

        faces = []
        locations = []
        predictions = []

        # Process all detected faces first
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # Filter out weak detections
            if confidence > 0.5: # Face detection confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                if face.size == 0: # Skip empty faces
                    continue

                # Preprocessing for Keras mask classifier
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = np.expand_dims(face, axis=0) # Add batch dimension
                face = face / 255.0 # Normalize

                faces.append(face)
                locations.append((startX, startY, endX, endY))
        

        # Initialize counters for the dashboard
        mask_count = 0
        no_mask_count = 0

        # Loop over the locations and their corresponding predictions to draw and count
        for (box, pred) in zip(locations, predictions):
            (startX, startY, endX, endY) = box
            mask_prob = pred[0] # The model outputs a single value due to sigmoid

            if mask_prob > 0.5: # Mask classification confidence threshold
                label = "Mask"
                color = (0, 255, 0) # Green for Mask
                mask_count += 1
            else:
                label = "No Mask"
                color = (0, 0, 255) # Red for No Mask
                no_mask_count += 1
                # Log violation to database
                log_violation(label)
                # Emit event for real-time log display on dashboard
                socketio.emit('log_event', {'message': f'Violation: No Mask Detected at {datetime.now().strftime("%H:%M:%S")}!'})

            # Format label for display on frame
            label_text = f"{label}: {mask_prob * 100:.2f}%" # Show confidence as 0.00%
            cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # --- Emit Data to Frontend ---
        # 1. Update general statistics
        socketio.emit('update_stats', {
            'total_faces': len(faces),
            'mask_count': mask_count,
            'no_mask_count': no_mask_count
        })

        # 2. Stream the video frame (base64 encoded)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'image': frame_b64})

        # Yield control to the eventlet server to prevent blocking
        socketio.sleep(0.01) # Small sleep to allow other events to be processed

    video_capture.release()
    print("[INFO] Video processing thread stopped.")

# --- Flask Routes and SocketIO Events ---
@app.route('/')
def index():
    """Serves the main dashboard page."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handles a new client connection. Starts the background video processing thread if not already running."""
    global video_thread
    with thread_lock:
        if video_thread is None:
            video_thread = socketio.start_background_task(target=video_processing_thread)
    print("Client connected.")

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnection."""
    print("Client disconnected.")

# --- API Endpoint for Historical Violation Statistics ---
@app.route('/api/violations_history')
def api_violations_history():
    """
    API endpoint to fetch all historical mask violation records from the database.
    Returns data as JSON for frontend display.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT label, timestamp FROM violations ORDER BY timestamp DESC")
        records = cursor.fetchall()
        conn.close()

        # Format data for JSON response
        formatted_records = []
        for label, timestamp_str in records:
            dt_object = datetime.fromisoformat(timestamp_str)
            formatted_records.append({
                'label': label,
                'timestamp': dt_object.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Optional: Add summary counts to the history endpoint as well
        total_violations_count = len(records)
        no_mask_violations_count = sum(1 for r in records if r[0] == 'No Mask')
        
        # Count by date for daily summary
        date_counts = Counter(datetime.fromisoformat(r[1]).strftime('%Y-%m-%d') for r in records)
        sorted_date_counts = dict(sorted(date_counts.items()))

        return jsonify({
            'history': formatted_records,
            'summary': {
                'total_logged_incidents': total_violations_count,
                'total_no_mask_violations': no_mask_violations_count,
                'violations_by_date': sorted_date_counts
            }
        })

    except Exception as e:
        print(f"❌ Error fetching historical violations: {e}")
        return jsonify({'error': str(e)}), 500


# --- Main Execution ---
if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    # Use eventlet as the web server for performance with SocketIO
    # Host '0.0.0.0' makes it accessible on your network (useful for testing on other devices)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
>>>>>>> de9d79f66712955c59bdc08515cf835fe91d4454
