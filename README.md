# DetectCarDistanceAndRoadLane
Project that estimates the distance a car is on a road based on the relationship between the real size of the car and the size it appears in the video obtained.
It also estimates the lane you are traveling in at any given time based on the angle between the position of the car and the video camera, and you can even guess lane change intentions based on this angle. Also included is a test using the new distance calculation functions with Ultralytics.

It is an attempt to apply the project https://medium.com/@batuhansenerr/object-distance-direction-detection-for-blind-and-low-vision-people-c3f3fd83cbef to the case of detection of distances of cars and traffic lanes roads, despite the differences between both cases.

A standard car width is assumed equal to the width of a parking space (2.3 meters).
To avoid the dispersion of the angles with which the lane is estimated, it is only calculated at a distance of 15 meters from the camera

Requirements: all the files that accompany the project and packages that, if missing, can be installed with a simple pip

import numpy as np

import cv2

import time

import math

from ultralytics import YOLO

It is tested by running:

VIDEODetectCarDistanceAndRoadLane.py

best.pt is the model to detect the cars in the video and was created following the instructions that appear in the project 
https://github.com/ablanco1950/LicensePlate_Yolov8_MaxFilters training yolo with a custom roboflow dataset

Another test can be performed by running the program

CarsDistances.py

Which is an adaptation and simplification of the project:

https://github.com/VforVitorio/F1_AI_team_detection/blob/main/YOLO_fine_tune.ipynb

In this case, the calculation of distances from a car to other cars in its vicinity is simulated.
<video src='https://github.com/ablanco1950/DetectCarDistanceAndRoadLane/blob/main/ReducedCarsDistances_output.mp4' width=180/>


https://github.com/user-attachments/assets/46ad1c43-1996-44c6-8828-4037cf745e21




Good results seem to be obtained by running: DetectCarDistance_with_ultralytics.py

adaptation and simplification from https://medium.com/@VK_Venkatkumar/video-analytics-counting-speed-distance-estimation-with-graph-visualization-yolov10-da1c24f7f245
which also estimates the speed

For that you must have an upgraded version of ultralytics and the proper version of lap:

inside conda in the scripts directory of the user environment

python pip-script.py install --no-cache-dir "lapx>=0.5.2"

upgrade ultralytics

python pip-script.py install --upgrade ultralytics

References:

  https://medium.com/@batuhansenerr/object-distance-direction-detection-for-blind-and-low-vision-people-c3f3fd83cbef


The video used: Traffic IP Camera video.mp4 test video, was  downloaded from https://github.com/anmspro/Traffic-Signal-Violation-Detection-System/tree/master/Resources

https://github.com/VforVitorio/F1_AI_team_detection/blob/main/YOLO_fine_tune.ipynb

https://medium.com/@VforVitorio/decoding-the-grid-teaching-ai-to-see-formula-1-5c3018011811
              
https://stackoverflow.com/questions/26896837/relationship-between-the-size-of-the-object-and-the-distance-to-the-camera

https://stackoverflow.com/questions/14038002/opencv-how-to-calculate-distance-between-camera-and-object-using-image

https://www.todoexpertos.com/categorias/ciencias-e-ingenieria/fisica/respuestas/1355998/tamano-de-los-objetos-con-la-distancia

https://es.planetcalc.com/1897/

https://photo.stackexchange.com/questions/40981/what-is-the-relationship-between-size-of-object-with-distance

https://github.com/ablanco1950/LicensePlate_Yolov8_MaxFilters

https://medium.com/@VK_Venkatkumar/video-analytics-counting-speed-distance-estimation-with-graph-visualization-yolov10-da1c24f7f245


