import cv2
import numpy as np
from ultralytics import YOLO

# ======================
# CONFIG
# ======================
VIDEO_PATH = r"D:\learn\SIH\TrafficAI\traffic_test.mp4"

default_model = YOLO("yolov8n.pt")  
ambulance_model = YOLO(r"D:/learn/SIH/TrafficAI/notebooks/runs/detect/ambulance_model/weights/best.pt")

vehicle_classes = [2, 3, 5, 7]  
ambulance_keywords = ["ambulance"]

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    print("❌ Error: Cannot open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete.")
        break

    h_orig, w_orig = frame.shape[:2]
    display_frame = frame.copy()

    # --------------------------
    # ZOOM OUT effect (optional)
    # --------------------------
    zoom_scale = 0.85  # 1.0 = no zoom, <1 = zoom out
    new_w, new_h = int(w_orig * zoom_scale), int(h_orig * zoom_scale)
    resized = cv2.resize(display_frame, (new_w, new_h))
    # Pad to original size
    pad_x = (w_orig - new_w) // 2
    pad_y = (h_orig - new_h) // 2
    display_frame = cv2.copyMakeBorder(resized, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # --------------------------
    # Run YOLO detection
    # --------------------------
    resized_for_yolo = cv2.resize(frame, (1280, 1280))
    results_default = default_model.predict(resized_for_yolo, conf=0.4, verbose=False)
    results_ambulance = ambulance_model.predict(resized_for_yolo, conf=0.4, verbose=False)

    scale_x, scale_y = w_orig / 1280, h_orig / 1280

    vehicle_count = 0
    emergency_detected = False

    # Draw vehicles
    for d in results_default[0].boxes.data:
        x1, y1, x2, y2, conf, cls = d
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        class_id = int(cls)
        label = results_default[0].names[class_id].lower()

        if class_id in vehicle_classes:
            vehicle_count += 1
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw ambulances
    for d in results_ambulance[0].boxes.data:
        x1, y1, x2, y2, conf, cls = d
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        label = results_ambulance[0].names[int(cls)].lower()

        if any(keyword in label for keyword in ambulance_keywords):
            emergency_detected = True
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --------------------------
    # Traffic light logic
    # --------------------------
    if emergency_detected:
        green_time = 30
        light_status = "🚨 Emergency! Green"
        active_light = "green"
    elif vehicle_count < 5:
        green_time = 5
        light_status = "🔴 Red"
        active_light = "red"
    elif vehicle_count < 15:
        green_time = 10
        light_status = "🟢 Green"
        active_light = "green"
    else:
        green_time = 20
        light_status = "🟢 Green"
        active_light = "green"

    # Info text
    cv2.putText(display_frame, f'Vehicles: {vehicle_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display_frame, f'Light: {light_status} ({green_time}s)', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # --------------------------
    # Draw traffic signal always visible
    # --------------------------
    light_x, light_y = 20, 120
    cv2.rectangle(display_frame, (light_x, light_y), (light_x + 50, light_y + 150), (50, 50, 50), -1)

    colors = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0)
    }

    cv2.circle(display_frame, (light_x + 25, light_y + 25), 15,
               colors["red"] if active_light == "red" else (100, 100, 100), -1)
    cv2.circle(display_frame, (light_x + 25, light_y + 75), 15,
               colors["yellow"] if active_light == "yellow" else (100, 100, 100), -1)
    cv2.circle(display_frame, (light_x + 25, light_y + 125), 15,
               colors["green"] if active_light == "green" else (100, 100, 100), -1)

    # --------------------------
    # Show video
    # --------------------------
    cv2.imshow("🚦 Traffic AI - Zoom Out + Traffic Animation", display_frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
