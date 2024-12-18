from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import datetime
from urllib.parse import quote_plus
from bson import ObjectId

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# MongoDB Setup
username = quote_plus("dustbin")  # Replace with your MongoDB username
password = quote_plus("Dustbin@123")  # Replace with your MongoDB password
MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.6isqu.mongodb.net/"

DATABASE_NAME = "garbage_detection"
COLLECTION_NAME = "final_data"

# Connect to MongoDB
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DATABASE_NAME]
    detections_collection = db['detection']  # 'detection' collection
    sensor_collection = db[COLLECTION_NAME]
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# Helper function to convert ObjectId to string
def convert_objectid(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    return obj

@app.route('/final_result', methods=['GET'])  # Corrected route with a slash
def final_result():
    try:
        # Fetch detection data and sensor data
        detection_data = list(detections_collection.find())
        sensor_data = list(sensor_collection.find())

        final_results = []
        camera_better_count = 0
        sensor_better_count = 0
        both_agree_count = 0

        # Iterate through each detection record
        for det in detection_data:
            det_time = datetime.datetime.strptime(det['timestamp'], "%Y-%m-%d %H:%M:%S")
            det_class = det['classification']  # Detected classification (dry or wet)
            det_accuracy = det['accuracy']

            # Find closest sensor data based on timestamp
            matching_sensor = None
            for sensor in sensor_data:
                sensor_time = datetime.datetime.strptime(sensor['timestamp'], "%Y-%m-%d %H:%M:%S")
                if abs(det_time - sensor_time) <= datetime.timedelta(seconds=5):
                    matching_sensor = sensor
                    break

            if matching_sensor:
                # Extract sensor data
                humidity = matching_sensor['humidity']
                moisture = matching_sensor['moisture']
                dryness_percentage = matching_sensor['drynessPercentage']
                wetness_percentage = matching_sensor['wetnessPercentage']

                # Classification logic based on thresholds
                if humidity < 50 and moisture < 30:
                    sensor_class = "Dry"
                elif humidity >= 50 and moisture >= 30:
                    sensor_class = "Wet"
                else:
                    if dryness_percentage > wetness_percentage:
                        sensor_class = "Dry"
                    elif wetness_percentage > dryness_percentage:
                        sensor_class = "Wet"
                    else:
                        sensor_class = "Common"

                # Comparison logic
                if det_class == sensor_class:
                    both_agree_count += 1
                    final_results.append({
                        "timestamp": det['timestamp'],
                        "final_object": det_class,
                        "source": "Both",
                        "confidence": det_accuracy if det_accuracy >= 50 else "Sensor"
                    })
                else:
                    if det_accuracy >= 50:
                        camera_better_count += 1
                        final_results.append({
                            "timestamp": det['timestamp'],
                            "final_object": det_class,
                            "reason": "Camera confidence higher"
                        })
                    else:
                        sensor_better_count += 1
                        final_results.append({
                            "timestamp": det['timestamp'],
                            "final_object": sensor_class,
                            "reason": "Sensor classification considered more reliable"
                        })

        # Calculate statistics
        total_records = len(detection_data)
        result_summary = {
            "total_records": total_records,
            "camera_better": camera_better_count,
            "sensor_better": sensor_better_count,
            "both_agree": both_agree_count,
            "agreement_percentage": (both_agree_count / total_records * 100) if total_records > 0 else 0
        }

        # Return the final results with analysis summary
        return jsonify({
            "detection_data": [convert_objectid(det) for det in detection_data],
            "sensor_data": [convert_objectid(sensor) for sensor in sensor_data],
            "final_results": final_results,
            "result_summary": result_summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5002)
