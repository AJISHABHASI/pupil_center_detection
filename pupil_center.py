# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:24:01 2019

@author: Ajisha
"""

import numpy as np
import cv2
from skimage.measure import label, regionprops
cap = cv2.VideoCapture('D:\dataset\InputEye.mp4')

i=1
while(1):
    ret, frame = cap.read()
    if ret is False:
        break
    
    print(i)
    i=i+1
     
    
    image = frame

    cv2.imshow("Input", image)
    shifted=frame 
  
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Input", gray)
    
    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
    cl1 = clahe.apply(gray)
    gray=cv2.equalizeHist(cl1)
    
    
    
    thresh = cv2.threshold(gray, 1, 255,cv2.THRESH_BINARY)[1]
    
    kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    closing = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Thresh", closing)
    
    label_img = label(~closing)
    regions = regionprops(label_img)
    
    for props in regions:

        orientation = props.orientation
        if props.area>80:
            y0, x0 = props.centroid
            cv2.putText(frame,'+',(int(x0),int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 100), 1,cv2.LINE_AA )
            
        cv2.imshow("Output", frame)   


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
