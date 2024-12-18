from flask import Flask, Response
from ultralytics import YOLO
import cv2
import time
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsRegressor
from pymongo import MongoClient
from datetime import datetime, timezone
from urllib.parse import quote_plus
import requests 

# Initialize Flask app
app = Flask(__name__)

# MongoDB Setup
username = quote_plus("dustbin")  # MongoDB username
password = quote_plus("Dustbin@123")  # MongoDB password
MONGO_URI = f'mongodb+srv://{username}:{password}@cluster0.6isqu.mongodb.net/'
DATABASE_NAME = "garbage_detection"
COLLECTION_NAME = "detections"

# Connect to MongoDB and create TTL index for automatic deletion
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DATABASE_NAME]
    detections_collection = db[COLLECTION_NAME]
    detections_collection.create_index("timestamp", expireAfterSeconds=7200)  # TTL of 2 hours
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# Insert detection data into MongoDB
def insert_detection_data(garbage_model, predicted_distance, dry_wet_label, dry_wet_confidence, class_id, conf, dry_percent, wet_percent):
    timestamp = datetime.now(timezone.utc).isoformat()
    object_name = garbage_model.names[int(class_id)]
    if object_name == "Dry Waste":
        object_name = "Garbage"
    detection_data = {
        "object": object_name,
        "confidence": float(conf),
        "distance": float(predicted_distance),
        "dry_wet_label": dry_wet_label,
        "dry_wet_confidence": float(dry_wet_confidence),
        "dry_percent": dry_percent,
        "wet_percent": wet_percent,
        "timestamp": timestamp
    }
    try:
        detections_collection.insert_one(detection_data)
        print("Detection data saved to MongoDB.")
    except Exception as e:
        print(f"Error saving data to MongoDB: {e}")

# Load model paths from pickle file
try:
    with open('models_metadata.pkl', 'rb') as f:
        models = pickle.load(f)
except Exception as e:
    print(f"Error loading model metadata: {e}")
    exit()

# Load YOLO models
try:
    garbage_model = YOLO(models['garbage_model']['model_path'])
    dry_wet_model = YOLO(models['drywet_model']['model_path'])
except Exception as e:
    print(f"Error loading YOLO models: {e}")
    exit()

# KNN for distance estimation
bbox_size = np.array([[50, 50], [100, 100], [150, 150], [200, 200], [250, 250]])
distance = np.array([3, 2, 1.5, 1, 0.5])
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(bbox_size, distance)

# Define boundary parameters
boundary_distance = 1.0  # 1 meter

# ESP32-CAM stream URL
ESP32_CAM_URL = "http://192.168.1.100/cam-hi.jpg"  # Replace with your ESP32-CAM URL

# Frame settings
fps_limit = 90  # Frames per second limit
last_frame_time = 0

# Function to fetch frames from ESP32-CAM
def fetch_frame():
    try:
        response = requests.get(ESP32_CAM_URL, stream=True, timeout=5)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error fetching frame: {e}")
    return None

# Function to draw a curved boundary line
def draw_curved_boundary(frame):
    frame_height, frame_width = frame.shape[:2]
    curve_depth = 30
    curve_center_y = frame_height // 2
    num_points = frame_width
    curve_points = [(x, curve_center_y + int(curve_depth * np.sin(np.pi * x / (frame_width - 1)))) for x in range(num_points)]
    for i in range(len(curve_points) - 1):
        cv2.line(frame, curve_points[i], curve_points[i + 1], (0, 255, 0), 2)
    return frame

# Function to process and generate video feed
def generate_video():
    global last_frame_time
    while True:
        current_time = time.time()
        if current_time - last_frame_time < 1 / fps_limit:
            time.sleep(0.01)
            continue
        frame = fetch_frame()
        if frame is None:
            print("Failed to fetch frame from ESP32-CAM.")
            time.sleep(0.5)
            continue
        garbage_results = garbage_model(frame)
        for result in garbage_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                class_id = box.cls[0]
                if conf < 0.5:
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f'{garbage_model.names[int(class_id)]} {conf:.2f}', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cropped_garbage = frame[y1:y2, x1:x2]
                dry_wet_results = dry_wet_model(cropped_garbage)
                dry_wet_label = 'Common'
                dry_wet_confidence, dry_percent, wet_percent = 0.0, 0.0, 0.0
                for dw_result in dry_wet_results:
                    if len(dw_result.boxes) > 0:
                        dw_class_id = dw_result.boxes[0].cls[0]
                        dry_wet_label = dry_wet_model.names[int(dw_class_id)]
                        dry_wet_confidence = dw_result.boxes[0].conf[0]
                        wet_confidence = 1.0 - dry_wet_confidence
                        dry_percent = dry_wet_confidence * 100
                        wet_percent = wet_confidence * 100
                cv2.putText(frame, f"Dry: {dry_percent:.2f}%, Wet: {wet_percent:.2f}%", 
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                box_width, box_height = x2 - x1, y2 - y1
                predicted_distance = knn.predict([[box_width, box_height]])[0]
                boundary_text = f"Inside 1m ({predicted_distance:.2f}m)" if predicted_distance <= boundary_distance else f"Outside 1m ({predicted_distance:.2f}m)"
                boundary_color = (0, 255, 0) if predicted_distance <= boundary_distance else (0, 0, 255)
                cv2.putText(frame, boundary_text, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, boundary_color, 2)
                insert_detection_data(garbage_model, predicted_distance, dry_wet_label, dry_wet_confidence, class_id, conf, dry_percent, wet_percent)
        frame = draw_curved_boundary(frame)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# Video stream route
@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
