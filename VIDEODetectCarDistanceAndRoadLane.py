# -*- coding: utf-8 -*-
"""
@author:  Alfonso Blanco

Adapting
 https://medium.com/@batuhansenerr/object-distance-direction-detection-for-blind-and-low-vision-people-c3f3fd83cbef

"""
######################################################################
# PARAMETERS
#####################################################################
######################################################
# Video from https://github.com/anmspro/Traffic-Signal-Violation-Detection-System/tree/master/Resources
dirVideo="Traffic IP Camera video.mp4"

# in  14 minutes = 800 seconds finish  
TimeLimit=800
# Max number of Snapshots to consider a image
LimitSnapshot=1
# Max number of Snapshots to consider a image
# lower 3 snapshots is consider noisy

# to increase the speed of the process,
# even if some license plates are lost,
# only one snapshot out of every SpeedUpFrames is processed
SpeedUpFrames=5

# to increase speed, jump frames  
ContFramesJumped=0
fps=25 #frames per second of video dirvideo, see its properties
fpsReal= fps/SpeedUpFrames # To speed up the process only one of SpeedUpFrames
                           # is considered
lenthRegion=4.5 #the depth of the considered region corresponds
                # to the length of a parking space which is usually 4.5m                           
##############################################################
# DOWNLOAD VIDEOS TO TEST
###############################################################
# video from https://github.com/hasaan21/Car-Number-Plate-Recognition-Sysytem
#dirVideo="vid.mp4"

#dirVideo="video12.mp4"
#dirVideo="C:\\Car_Speed_Detection\\Comma.ai.Data.and.Model\\Comma.ai Model\\train.mp4"

# from https://www.pexels.com/video/video-of-famous-landmark-on-a-city-during-daytime-1721294/
#dirVideo="Pexels Videos 1721294.mp4"

import numpy as np

import cv2

import time

import math

TimeIni=time.time()

dirnameYolo="best.pt"
# https://docs.ultralytics.com/python/
from ultralytics import YOLO
model = YOLO(dirnameYolo)
class_list = model.model.names
#print(class_list)




# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def DetectCarWithYolov8 (img):
  
    TabcropCar=[]
    y=[]
    yMax=[]
    x=[]
    xMax=[]
    results = model.predict(img)
    for i in range(len(results)):
        # may be several cars in a frame
        result=results[i]
        
        xyxy= result.boxes.xyxy.numpy()
        confidence= result.boxes.conf.numpy()
        class_id= result.boxes.cls.numpy().astype(int)
        # Get Class name
        class_name = [class_list[z] for z in class_id]
        # Pack together for easy use
        sum_output = list(zip(class_name, confidence,xyxy))
        # Copy image, in case that we need original image for something
        out_image = img.copy()
        for run_output in sum_output :
            # Unpack
            #print(class_name)
            label, con, box = run_output
            if label != "vehicle":continue
            cropCar=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
            
            TabcropCar.append(cropCar)
            y.append(int(box[1]))
            yMax.append(int(box[3]))
            x.append(int(box[0]))
            xMax.append(int(box[2]))
        
    return TabcropCar, y,yMax,x,xMax


###########################################################
# MAIN
##########################################################

TabLicensesmax=[]
ContLicensesmax=[]
TimeIniLicensesmax=[]
TimeEndLicensesmax=[]

ContDetected=0
ContNoDetected=0
TotHits=0
TotFailures=0

# A standard car width is assumed equal to the width of a parking space
# 2.3 meters
real_width=2.3

#with open( "VIDEOLicenseResults.txt" ,"w") as   w:
cap = cv2.VideoCapture(dirVideo)
     # https://levelup.gitconnected.com/opencv-python-reading-and-writing-images-and-videos-ed01669c660c

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps=5.0
frame_width = 680
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
     
video_writer = cv2.VideoWriter('demonstration.mp4',fourcc,fps, size)   
while (cap.isOpened()):
        ret, imgComplete = cap.read()
     
        if ret != True: break
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
        center_x = int(frame_width // 2)
        center_y = int(frame_height // 2)
        radius = min(center_x, center_y) - 30  # Radius of the circle where clock hands are drawn
        
        

        
        # speed up a little
            
        ContFramesJumped=ContFramesJumped+1
        if ContFramesJumped < SpeedUpFrames:
               continue
        else:
               ContFramesJumped=0        
           
           
        TabImgSelect, y1, y2, x1, x2 =DetectCarWithYolov8(imgComplete)
            
        if TabImgSelect==[]:
                #print(License + " NON DETECTED")
                ContNoDetected=ContNoDetected+1 
                continue
        else:
                ContDetected=ContDetected+1
                #print(License + " DETECTED ")
        for i in range(len(TabImgSelect)):
                img=TabImgSelect[i]  
                """
                Calculate the distance from real size of car and size in video
                from:
                https://medium.com/@batuhansenerr/object-distance-direction-detection-for-blind-and-low-vision-people-c3f3fd83cbef

                other references.
                https://stackoverflow.com/questions/26896837/relationship-between-the-size-of-the-object-and-the-distance-to-the-camera
                https://stackoverflow.com/questions/14038002/opencv-how-to-calculate-distance-between-camera-and-object-using-image
                https://www.todoexpertos.com/categorias/ciencias-e-ingenieria/fisica/respuestas/1355998/tamano-de-los-objetos-con-la-distancia
                https://es.planetcalc.com/1897/
                https://photo.stackexchange.com/questions/40981/what-is-the-relationship-between-size-of-object-with-distance


                """
                camera_width = x2[i] - x1[i]
                distance = (real_width * frame_width) / camera_width

                
                obj_center_x = (x1[i] + x2[i]) // 2
                obj_center_y = (y1[i] + y2[i]) // 2

                camera_middle_x = frame_width // 2
                camera_middle_y = frame_height // 2

                vector_x = obj_center_x - camera_middle_x
                vector_y = obj_center_y - camera_middle_y

                angle_deg = math.degrees(math.atan2(vector_y, vector_x))
                #direction = ''
                if angle_deg < 0:
                        angle_deg += 360
                        
                if 0 <= angle_deg < 5:
                        direction = "Right Lane"
                elif 5 <= angle_deg < 20:
                        direction = "Right Lane"
                elif 20 <= angle_deg < 60:
                        direction = "Center Lane"
                elif 60 <= angle_deg < 90:
                        direction = "Center Lane"
                elif 90 <= angle_deg < 120:
                        direction = "Center Lane"
                elif 120 <= angle_deg < 150:
                        direction = "Center Lane"
                elif 150 <= angle_deg < 180:
                        direction = "Left Lane"
                elif 180 <= angle_deg < 210:
                        direction = "Left Lane"
                elif 210 <= angle_deg < 240:
                        direction = "Left Lane"
                elif 240 <= angle_deg < 270:
                        direction = "Out Lane"
                elif 270 <= angle_deg < 300:
                        direction = "Out Lane"
                elif 300 <= angle_deg < 330:
                        direction = "Center Lane"
                elif 330 <= angle_deg < 336:
                        direction = "Center-Right Lane"
                elif 336 <= angle_deg < 352:
                        direction = "Center-Right Lane"         
                elif 352 <= angle_deg < 360:
                        direction = "Right Lane"        
                else:
                        direction = "OUT LANE"

                #cv2.putText(img, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #cv2.putText(imgComplete,direction + str(angle_deg) +  " Distance: {:.2f} meters".format(distance), (x1[i], y1[i] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # road lane detection by angle only accuracy in short distance
                if distance > 15:
                     cv2.putText(imgComplete,  " Distance: {:.2f} meters".format(distance), (x1[i], y1[i] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                     # to adjust the lane in function of angle
                     #cv2.putText(imgComplete,direction + str(angle_deg) , (x1[i], y1[i] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                     cv2.putText(imgComplete,direction +   " Distance: {:.2f} meters".format(distance), (x1[i], y1[i] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                #cv2.putText(imgComplete,direction + " " + str(angle_deg), (x1[i], y1[i] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(imgComplete, (x1[i], y1[i]), (x2[i], y2[i]), (255, 0, 255), 3)
                cv2.imshow("Webcam", imgComplete)
                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'): break 
                # saving video
                video_writer.write(imgComplete)    
                # a los 10 minutos = 600 segundos acaba     
                if time.time() - TimeIni > TimeLimit:       
                    break
                
                    
cap.release()
video_writer.release()
cv2.destroyAllWindows()
#w.close()    
     
    
print("")           
