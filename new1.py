import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from djitellopy import Tello
import time
import warnings
import logging
import sys

# === Suppress YOLO logs ===
logging.getLogger("ultralytics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
sys.stdout = open('nul', 'w') if sys.platform == 'win32' else open('/dev/null', 'w')
yolo_model = YOLO("best.pt")  # Replace with your model path
sys.stdout = sys.__stdout__

# === Init Tracker and Drone ===
tracker = DeepSort(max_age=30)

tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()
tello.takeoff()
tello.send_rc_control(0, 0, -20, 0)
time.sleep(2)
# === PID Setup ===
pid = [0.4, 0.4, 0]
fbRange = [6200, 6800]  # Area thresholds for forward/back movement
pError = 0

# === Helper for Movement ===
def trackObject(info, frame_w, pid, pError):
    area = info[1]
    x = info[0][0]
    fb = 0

    error = x - frame_w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    if fbRange[0] < area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20

    if x == 0:
        speed = 0
        error = 0

    tello.send_rc_control(0, fb, 0, speed)
    return error

# === Main Loop ===
frame_read = tello.get_frame_read()
cv2.namedWindow("Object Tracker", cv2.WINDOW_NORMAL)

while True:
    frame = frame_read.frame
    if frame is None:
        continue

    frame = cv2.resize(frame, (640, 480))
    display_frame = frame.copy()

    # === YOLOv8 Detection ===
    results = yolo_model.predict(frame, conf=0.7, iou=0.7, imgsz=640, max_det=1, verbose=False)

    detections = []
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)

        # PID-based movement
        pError = trackObject([[cx, cy], area], frame.shape[1], pid, pError)

        # Draw visuals
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        cv2.putText(display_frame, f'ID: {track.track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("Object Tracker", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.land()
        break

tello.streamoff()
cv2.destroyAllWindows()
