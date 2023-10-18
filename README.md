# DetectCarDistanceAndRoadLane
Project that estimates the distance a car is on a road based on the relationship between the real size of the car and the size it appears in the video obtained.
It also estimates the lane you are traveling in at any given time based on the angle between the position of the car and the video camera, and you can even guess lane change intentions based on this angle.

It is an attempt to apply the project https://medium.com/@batuhansenerr/object-distance-direction-detection-for-blind-and-low-vision-people-c3f3fd83cbef to the case of detection of distances of cars and traffic lanes. roads, despite the differences between both cases.

A standard car width is assumed equal to the width of a parking space

Requirements: all the files that accompany the project and packages that, if missing, can be installed with a simple pip

import numpy as np

import cv2

import time

import math

It is tested by running:

VIDEODetectCarDistanceAndRoadLane.py

References:

  https://medium.com/@batuhansenerr/object-distance-direction-detection-for-blind-and-low-vision-people-c3f3fd83cbef
              
https://stackoverflow.com/questions/26896837/relationship-between-the-size-of-the-object-and-the-distance-to-the-camera

https://stackoverflow.com/questions/14038002/opencv-how-to-calculate-distance-between-camera-and-object-using-image

https://www.todoexpertos.com/categorias/ciencias-e-ingenieria/fisica/respuestas/1355998/tamano-de-los-objetos-con-la-distancia

https://es.planetcalc.com/1897/

https://photo.stackexchange.com/questions/40981/what-is-the-relationship-between-size-of-object-with-distance
