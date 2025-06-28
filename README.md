# myIBMproject
This is my current project for IBM internship which is related to "Face Mask Detection"


# 😷 Face Mask Detection App

This project is a real-time face mask detection web app built using:

- **YOLOv8** for object detection (face)
- **MediaPipe** for landmark validation
- **TensorFlow/Keras** (optional) for mask classification
- **Flask** for web server
- **SQLite** for logging mask violations
- **HTML/CSS/JavaScript** for frontend interface

---

## 🔥 Features

- Real-time camera access
- Detects multiple faces at once
- Classifies as `Mask` 😷 or `No Mask` ❌
- Highlights detections with bounding boxes
- Stores timestamped violations in a SQLite database
- Dashboard to view live feed and logs

---

## 🗂️ Project Structure

![image](https://github.com/user-attachments/assets/a5d84c2d-57fa-433f-bbbf-7a0e4eede49e)


---

## 🚀 How to Run

1.clone the repo 
 git clone https://github.com/your-username/IBMproject.git

2.Install Requirements 
pip install -r requirements.txt

3.Run the app
python app.py

4.Open in browser


🧠 Model Info
The YOLOv8 model is trained on a custom dataset yolo_dataset_ready.

The MediaPipe integration ensures detections are valid human faces (avoids false positives).

🗃️ Database
mask_violations.db stores:

Timestamp

Detection type (mask or no_mask)

Image (optional as base64 or file path)



📦 Dataset
Located in yolo_dataset_ready/ folder

Label format: YOLO annotation (.txt files)


🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first.




