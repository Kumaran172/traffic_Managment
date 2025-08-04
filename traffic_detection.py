import cv2
from ultralytics import YOLO

# Load models
default_model = YOLO("yolov8n.pt")  # Pretrained COCO model for general vehicles
ambulance_model = YOLO("D:/learn/SIH/TrafficAI/notebooks/runs/detect/ambulance_model/weights/best.pt")  # Custom-trained ambulance detector

# Class IDs for common vehicles (COCO classes)
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
ambulance_keywords = ["ambulance"]

# Start video capture
cap = cv2.VideoCapture("http://10.169.197.126:8080/video")  # Or replace with your IP cam URL

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run both models
    results_default = default_model(frame, verbose=False)
    results_ambulance = ambulance_model(frame, verbose=False)

    detections_default = results_default[0].boxes.data
    detections_ambulance = results_ambulance[0].boxes.data

    vehicle_count = 0
    emergency_detected = False

    # Draw general vehicles
    for d in detections_default:
        x1, y1, x2, y2, conf, cls = d
        class_id = int(cls)
        label = results_default[0].names[class_id].lower()

        if class_id in vehicle_classes:
            color = (0, 255, 0)  # Green box
            vehicle_count += 1
        else:
            continue

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw ambulances from custom model
    for d in detections_ambulance:
        x1, y1, x2, y2, conf, cls = d
        label = results_ambulance[0].names[int(cls)].lower()

        if any(keyword in label for keyword in ambulance_keywords):
            color = (0, 0, 255)  # Red box for ambulance
            emergency_detected = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Traffic light logic
    if emergency_detected:
        green_time = 30
        light_color = (0, 255, 0)
        light_status = "Emergency! Green"
    elif vehicle_count < 5:
        green_time = 5
        light_color = (0, 0, 255)
        light_status = "Red"
    elif vehicle_count < 15:
        green_time = 10
        light_color = (0, 255, 0)
        light_status = "Green"
    else:
        green_time = 20
        light_color = (0, 255, 0)
        light_status = "Green"

    # Draw info
    cv2.putText(frame, f'Vehicles: {vehicle_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Light: {light_status} ({green_time}s)', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Traffic light icon
    cv2.rectangle(frame, (frame.shape[1] - 100, 30), (frame.shape[1] - 50, 130), (50, 50, 50), -1)
    cv2.circle(frame, (frame.shape[1] - 75, 80), 20, light_color, -1)

    # Display frame
    cv2.imshow("Traffic AI", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
