# https://medium.com/@VK_Venkatkumar/video-analytics-counting-speed-distance-estimation-with-graph-visualization-yolov10-da1c24f7f245

import cv2
import math
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.solutions import speed_estimation

# Initialize YOLO models
object_detection_model = YOLO("yolov8s.pt")
speed_estimation_model = YOLO("yolov8n.pt")
names = speed_estimation_model.model.names
#print(names)

# Open video file

cap = cv2.VideoCapture("Traffic IP Camera video.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


# Initialize video writer

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('demonstration.mp4',fourcc,fps, size)  

frame_count = 0
data = {}
labels = []
class_counts_over_time = {}
speed_over_time = {}
distance_over_time = {}

# Center point and pixel per meter for distance calculation
#center_point = (0, h)
center_point = (int(w/2), h)

# NEED TO BE ADJUSTED
# https://stackoverflow.com/questions/50408347/how-to-find-pixel-per-meter
pixel_per_meter = 10

# Line points for speed estimation
#line_pts = [(0, 360), (1280, 360)]
#line_pts = [(0, h), (w, h)]
line_pts = [(0, 485), (1280, 485)]

# Initialize speed-estimation object
speed_obj = speed_estimation.SpeedEstimator(names=names, reg_pts=line_pts, view_img=False)


# Colors for text and bounding box
txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))


desired_classes = {'car': 2}
while cap.isOpened():
    
    # Modified to adapt video
    fps=5.0
    frame_width = 680
    frame_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # Object detection for speed estimation
    speed_tracks = speed_estimation_model.track(frame, persist=True, show=False)
    frame = speed_obj.estimate_speed(frame, speed_tracks)

    # Object detection for distance estimation
    annotator = Annotator(frame, line_width=2)
    results = object_detection_model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            cls_name = object_detection_model.names[int(cls)]
            if cls_name in desired_classes and desired_classes[cls_name] == cls:  # Filter desired classes and IDs
                
                annotator.box_label(box, label=str(track_id), color=bbox_clr)
                annotator.visioneye(box, center_point)

                x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)  # Bounding box centroid

                distance = (math.sqrt((x1 - center_point[0]) ** 2 + (y1 - center_point[1]) ** 2)) / pixel_per_meter

                text_size, _ = cv2.getTextSize(f"Distance: {distance:.2f} m", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), txt_background, -1)
                cv2.putText(frame, f"Distance: {distance:.2f} m", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, txt_color, 3)

                if track_id not in distance_over_time:
                    distance_over_time[track_id] = [0] * (frame_count - 1)
                distance_over_time[track_id].append(distance)

               
                speed = speed_obj.speeds.get(track_id, 0) if hasattr(speed_obj, 'speeds') else 0
                if track_id not in speed_over_time:
                    speed_over_time[track_id] = [0] * (frame_count - 1)
                speed_over_time[track_id].append(speed)

   
    cv2.imshow("Webcam", frame)
    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'): break 


    # Write the frame with visualizations
    out.write(frame)

   
cap.release()
out.release()
cv2.destroyAllWindows()

