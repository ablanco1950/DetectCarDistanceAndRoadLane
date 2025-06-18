# 
# Adapted from:
# https://github.com/VforVitorio/F1_AI_team_detection/blob/main/YOLO_fine_tune.ipynb
#https://medium.com/@VforVitorio/decoding-the-grid-teaching-ai-to-see-formula-1-5c3018011811
# https://medium.com/@ahsaanraazaa/yolov8-and-opencv-quick-start-guide-c5603111ed56

dirvideo="project_video.mp4"
#dirvideo="Traffic IP Camera video.mp4"

from ultralytics import YOLO

import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter
import time as pytime

FRAME_WIDTH = 1280
CAR_LENGTH_METERS = 5.63  # Real car length in meters
# FOR GAPS: use a lower global threshold to maximize detections
GAP_DETECTION_THRESHOLD = 0.40  # Low threshold to detect all possible cars


def calculate_gap(box1, box2):    
    """Calculates the distance between centers using car width for scale"""
    # Box centers
    cx1, cy1 = (box1[0] + box1[2])/2, (box1[1] + box1[3])/2
    cx2, cy2 = (box2[0] + box2[2])/2, (box2[1] + box2[3])/2
    
    # Distance in pixels
    pixel_distance = np.hypot(cx2 - cx1, cy2 - cy1)
    
    # Scale based on average width of detected cars
    avg_width = ((box1[2] - box1[0]) + (box2[2] - box2[0])) / 2
    scale = CAR_LENGTH_METERS / avg_width if avg_width != 0 else 0
    
    # Calculate gap time at 120km/h (33.33 m/s)
    speed_mps = 33.33  # Meters per second at 120km/h
    gap_time = (pixel_distance * scale) / speed_mps
    
    return {
        'distance': pixel_distance * scale,  # Distance in meters
        'time': gap_time,                   # Time in seconds at 120km/h, suposing all cars about this speed
       
    }

def main():
    
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Load the YOLO12 model
    #model = YOLO('yolo12m.pt') # slower but more precise
    
    GAP_DETECTION_THRESHOLD = 0.40  # Low threshold to detect all possible cars
    class_names = model.names # Get class names (e.g., 'person', 'car')

    # Detection history for stabilization
    last_detections = {}
    track_history = {}
    id_counter = 0
    class_history = {}
    track_age= {}

    output_path = "CarsDistances_output.mp4"

    # Open the default webcam (0 for built-in, or specify video file path)   
    cap = cv2.VideoCapture(dirvideo)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    ## Get original video dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # Calculate new height maintaining aspect ratio
    target_height = int(FRAME_WIDTH * original_height / original_width)

    out = None
    if output_path:
        # Change codec from 'mp4v' to 'XVID' which is more reliable
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (FRAME_WIDTH, target_height))
        if not out.isOpened():
            print(f"Error: Could not create output video file at {output_path}")
            print("Continuing without saving output...")
            output_path = None

    # Variables for calculating real FPS
    frame_count = 0
    start_time = pytime.time()
    current_fps = 0 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or error reading frame.")
            break

        frame_resized = cv2.resize(frame, (FRAME_WIDTH, target_height))
        original_frame = frame_resized.copy()

        # Run inference on the frame
        results = model(frame, conf=0.5) # conf=0.5 means 50% confidence threshold

        # Current detections
        current_detections = {}

        # Process and draw detections
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy()

            # Create detection list with all information
            detections = []

            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if class_id != 2: continue # only cars
                cls_name=class_names[class_id]
                
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate center and size metrics
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                
                # Filter by size/proportion for problematic classes
                is_reasonable_size = area > (FRAME_WIDTH * target_height * 0.003)
                is_reasonable_ratio = 0.4 < aspect_ratio < 2.5
                
                
                
                # Assign unique ID or retrieve existing ID
                label = f"{class_names[int(class_id)]}: {conf:.2f}"
                # Assign unique ID or retrieve existing ID, only useful for overlaping detection ahead
                object_id = None
                for old_id, old_info in last_detections.items():
                    old_cx, old_cy = old_info['center']
                    old_cls = old_info['class']
                    
                    # Distance between centers
                    dist = np.sqrt((center_x - old_cx)**2 + (center_y - old_cy)**2)
                    
                    # If it's close, it could be the same object
                    if dist < 100:
                        object_id = old_id                        
                        break
                
                # If no match found, assign new ID
                if object_id is None:
                    object_id = id_counter
                    id_counter += 1
                    track_history[object_id] = []
                    class_history[object_id] = []
              
                # Update history
                if object_id in class_history:                    
                    class_history[object_id].append(cls_name)
                    # Limit history to 5 classes
                    if len(class_history[object_id]) > 5:
                        class_history[object_id].pop(0)
                
                    # Use the most common class from history for stability
                    if len(class_history[object_id]) >= 5:  # Increased from 3 to 5                        
                        counts = Counter(class_history[object_id])
                        if counts:  # Make sure it's not empty
                            most_common = counts.most_common(1)[0][0]
                            cls_name = most_common
                            classified = True
                
                # Save current detection
                current_detections[object_id] = {
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'class': cls_name,
                    'center': (center_x, center_y),
                    'area': area,
                    'y_bottom': y2  # For sorting by vertical position
                }
                
                # Add to detection list for gap calculation
                detections.append({
                    'id': object_id,
                    'box': (x1, y1, x2, y2),
                    'class': cls_name,
                    'conf': conf,
                    'y_bottom': y2
                })
            
            # Sort by vertical position (cars more below first - closer)
            detections = sorted(detections, key=lambda x: x['y_bottom'], reverse=True)
            
            # Eliminate ghost detections
            detections_to_keep = []
            for det in detections:
               
                    is_ghost = False    
                    for other in detections:
                        if other['id'] != det['id']:
                            # Calculate simple overlap between boxes
                            x1a, y1a, x2a, y2a = det['box']
                            x1b, y1b, x2b, y2b = other['box']
                            
                            overlap = max(0, min(x2a, x2b) - max(x1a, x1b)) * max(0, min(y2a, y2b) - max(y1a, y1b))
                            area_a = (x2a - x1a) * (y2a - y1a)
                            
                            # If there's significant overlap, likely a false positive
                            if overlap > 0.3 * area_a:
                                is_ghost = True
                                break
                    
                    if not is_ghost:
                        detections_to_keep.append(det)
                #else:
                #    detections_to_keep.append(det)
                    
            detections = detections_to_keep
            
            # Draw boxes and gaps
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['box']
                cls_name = det['class']
                conf = det['conf']
                color= (0,255,0)
                
                # Draw box with team color
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)               
                label = f"{cls_name}: {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame_resized, (x1, y1-t_size[1]-3), (x1+t_size[0], y1), color, -1)
                cv2.putText(frame_resized, label, (x1, y1-3), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                gap_info = calculate_gap(
                        det['box'], (int(FRAME_WIDTH/2),target_height,int(FRAME_WIDTH/2),target_height) 
                        
                    )
                 # Connection points
                cx1, cy1 = int((x1+x2)/2), int(y1)  # Use top of the car
                nx1, ny1, nx2, ny2 = (int(FRAME_WIDTH/2),target_height,int(FRAME_WIDTH/2),target_height)
                cx2, cy2 = int((nx1+nx2)/2), int(ny2)  # Use bottom of the next car
                    
                # Diagonal line between cars
                cv2.line(frame_resized, (cx1, cy1), (cx2, cy2), (0,0, 255), 2)
                    
                # Text at midpoint with more information
                mid_x, mid_y = (cx1+cx2)//2, (cy1+cy2)//2
                    
                # Distance and gap time
                dist_text = f"{gap_info['distance']:.1f}m"
                time_text = f"{gap_info['time']:.2f}s"
                    
                # Background for text
                dist_size = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                # Draw semi-transparent background
                overlay = frame_resized.copy()
                cv2.rectangle(overlay, 
                                 (mid_x - 5, mid_y - 50), 
                                 (mid_x + max(dist_size[0], time_size[0]) + 10, mid_y + 10),
                                 (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame_resized, 0.4, 0, frame_resized)
                    
                # Draw texts
                cv2.putText(frame_resized, dist_text, (mid_x, mid_y - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_resized, time_text, (mid_x, mid_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

               
        # Update last_detections for next iteration
        last_detections = current_detections
        
        # Calculate FPS
        if frame_count % 10 == 0:
            current_time = pytime.time()
            current_fps = 10.0 / (current_time - start_time)
            start_time = current_time
        
        # Show FPS and model information
        cv2.putText(frame_resized, f"FPS: {current_fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame_resized, "F1 Gap Detection", (FRAME_WIDTH - 300, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show detection mode and progress
        detection_mode = f"Detection Threshold: {GAP_DETECTION_THRESHOLD:.2f}"
        cv2.putText(frame_resized, detection_mode, (20, target_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show video progress
        progress_text = f"Frame: {current_frame}/{total_frames} ({current_frame/total_frames*100:.1f}%)"
        cv2.putText(frame_resized, progress_text, (FRAME_WIDTH - 400, target_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save processed frame if requested
        if output_path:
            out.write(frame_resized)
        
        # Show frame
        cv2.imshow("F1 Gap Detection", frame_resized)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('+'):  # Increase threshold
            GAP_DETECTION_THRESHOLD = min(GAP_DETECTION_THRESHOLD + 0.05, 0.95)
            print(f"Detection threshold increased to {GAP_DETECTION_THRESHOLD:.2f}")
        elif key == ord('-'):  # Decrease threshold
            GAP_DETECTION_THRESHOLD = max(GAP_DETECTION_THRESHOLD - 0.05, 0.05)
            print(f"Detection threshold decreased to {GAP_DETECTION_THRESHOLD:.2f}")
        elif key == ord('d'):  # Skip forward 10 seconds
            skip_frames = int(fps * 10)  # 10 seconds * fps = number of frames to skip
            new_frame_pos = min(current_frame + skip_frames, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_pos)
            current_frame = new_frame_pos - 1  # Will be incremented in the next cycle
            # Temporarily reset tracking
            last_detections = {}
            print(f"Skipped forward 10 seconds to frame {new_frame_pos}")
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()


   # cap.release()
   # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
