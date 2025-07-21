# REAL-TIME-PEOPLE-COUNTING-SYSTEM
PROJECT ON COUNTING PEOPLE IN A ROOM USING PYTHON

This project is used to count how many people are present in the room. 

**Backend for the project using python:**
1) YOLOv5 is used to detect people and rounding boxes are created around them. yolov5s is used version for its balance between speed and accuracy.

2) Centroid Tracker is used to track people coming inside the room and outside it. This algorithm calculates each person's centroid and unique ID is given to them. It tracks these IDs across frames by comparing centroids.This enables tracking of movements, ensuring each person is counted only once per crossing.

3) A vertical boundary line serves as a seperator between inside and outside the room. The system checks if a person’s centroid moves from one side of the line to the other, updating the count for total entries (Total IN) and exits (Total OUT).

**Frontend Display via OpenCV**:

1) Uses functions like cv2.imshow(), cv2.putText(), cv2.line(), and cv2.rectangle() to visually render:

    - Live video feed from the webcam

    - Bounding boxes (with rounded corners) around detected people

    - Arrows that illustrate movement direction

    - A transparent overlay displaying counters, IDs, and temporary entry/exit messages for 2 seconds

2) The window is set to fullscreen mode.
