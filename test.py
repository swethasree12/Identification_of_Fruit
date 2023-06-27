import numpy as np
import cv2  

frame = cv2.imread('images.jpg')
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 10, 120])
upper_red = np.array([15, 255, 255])

mask = cv2.inRange (hsv, lower_red, upper_red)
contours,temp = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if len(contours[i]) > 30:
        red_area = contours[i] #max(contours[i], key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(red_area)
        cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 0, 255), 2)
        region = frame[y:(y+h),x:(x+w)]
        cv2.imshow("aa",region)
        cv2.imwrite("extract/"+str(i)+".png",region)


cv2.imshow('frame', frame)
cv2.imshow('mask', mask)

cv2.waitKey(0)
