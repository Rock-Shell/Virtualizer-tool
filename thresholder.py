#object detection using hsv
# for video introduce cap variable

import cv2
import numpy as np

def nothing(x):
    pass

##cap=cv2.VideoCapture("ballmotionwhite.m4v")
##_,frame=cap.read()
frame=cv2.imread("room3.jpeg")
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# frame =
#  
#frame = np.zeros((512,512,3),np.uint8)
#frame[0:512,0:200] = (0,0,255)
#frame[0:512, 200:400] = (0,255,0)
#frame[0:512, 400:] = (255,0,0)

#cv2.imshow("f",frame)
# cv2.waitKey(0)
cv2.namedWindow('Tracking')
cv2.createTrackbar('LH','Tracking',0,255,nothing)
cv2.createTrackbar('LS','Tracking',0,255,nothing)
cv2.createTrackbar('LV','Tracking',0,255,nothing)
cv2.createTrackbar('UH','Tracking',255,255,nothing)
cv2.createTrackbar('US','Tracking',255,255,nothing)
# cv2.createTrackbar('UV','Tracking',255,255,nothing)

# cv2.setTrackbarPos("LH",'Tracking',60)

cv2.createTrackbar("gray_l","Tracking",0,255,nothing)
cv2.createTrackbar("gray_u","Tracking",255,255,nothing)

while True:
##    frame=cv2.imread('.png')
    #frame=cap.read()``
##    _, frame=cap.read()
    frame=cv2.resize(frame,(512,512))

    # hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    hsv = cv2.applyColorMap(gray, cv2.COLORMAP_HSV)

    l_h=cv2.getTrackbarPos('LH','Tracking')
    l_s=cv2.getTrackbarPos('LS','Tracking')
    l_v=cv2.getTrackbarPos('LV','Tracking')

    u_h=cv2.getTrackbarPos('UH','Tracking')
    u_s=cv2.getTrackbarPos('US','Tracking')
    # u_v=cv2.getTrackbarPos('UV','Tracking')

    gl=cv2.getTrackbarPos("gray_l","Tracking")
    gu=cv2.getTrackbarPos("gray_u","Tracking")

    #l_b=np.array([110,50,50])# lower bound of blue color on hsv
    #u_b=np.array([130,255,255])

    # l_b=np.array([l_h,l_s,l_v])
    # u_b=np.array([u_h,u_s,u_v])

    # mask=cv2.inRange(hsv,l_b,u_b)
    mask2=cv2.inRange(gray, gl,gu)
                #mask is data used for bitwise operation

    # res=cv2.bitwise_and(frame,frame, mask=mask)
    gr= cv2.bitwise_and(frame,frame, mask=mask2)

    cv2.imshow("img",frame)
    cv2.imshow("gray",gray)
    # cv2.imshow('hsv',hsv)
    # cv2.imshow('mask',mask)
    # cv2.imshow('res',res)
    cv2.imshow('gray_m',mask2)

    key=cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
