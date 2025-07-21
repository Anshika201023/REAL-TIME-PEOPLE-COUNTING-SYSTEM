import cv2
import torch
import numpy as np
import time
from centroid_tracker import CentroidTracker

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # confidence threshold

ct = CentroidTracker()
total_in, total_out = 0, 0
track_history = {}
entry_exit_log = {}  


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_x = frame_width // 2  # Center vertical line
cv2.namedWindow("People Counter", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("People Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    rects = []
    for *xyxy, conf, cls in detections:
        if int(cls) == 0:  # Class 0 is person
            x1, y1, x2, y2 = map(int, xyxy)
            rects.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 200, 250), 2, lineType=cv2.LINE_AA)

    objects = ct.update(rects)

    current_time = time.time()
    for objectID, centroid in objects.items():
        if objectID not in track_history:
            track_history[objectID] = centroid[0]

        prev_x = track_history[objectID]
        curr_x = centroid[0]
        cv2.arrowedLine(frame, (prev_x, centroid[1]), (curr_x, centroid[1]), (0, 255, 255), 2)

        # Entry / Exit Detection
        if objectID not in entry_exit_log:
            entry_exit_log[objectID] = {"last": 0, "message": ""}

        if prev_x < line_x and curr_x >= line_x:
            total_in += 1
            entry_exit_log[objectID] = {"last": current_time, "message": "Person Entered!"}
        elif prev_x > line_x and curr_x <= line_x:
            total_out += 1
            entry_exit_log[objectID] = {"last": current_time, "message": "Person Exited!"}

        track_history[objectID] = curr_x

        # ID label and center point
        cv2.putText(frame, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 160), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 2)

    cv2.putText(frame, f"Total IN: {total_in}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Total OUT: {total_out}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Currently Inside: {total_in - total_out}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    for objectID, data in entry_exit_log.items():
        if current_time - data["last"] <= 2:
            msg = data["message"]
            cv2.putText(frame, f"{msg} (ID {objectID})", (frame_width - 400, 50 + 30 * (objectID % 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
