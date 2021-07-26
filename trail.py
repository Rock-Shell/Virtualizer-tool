import cv2
import numpy as np

font=cv2.FONT_HERSHEY_SIMPLEX
draw = 0 # 0->no draw, 1-> +mask, 2-> -mask

def nothing(x):
        pass

def resize(img, size=(512,512)):
        img2 = cv2.resize(img,size)
        return img2

# grab the color of the wall and provide the lower and upper value
def click_event1(event, x, y , flags, param):
        global draw, mask, lb, ub
        if event==cv2.EVENT_LBUTTONDOWN:
                print(img[y,x], gray[y,x], hsv[y,x])
                lb = hsv[y,x]-10
                ub = hsv[y,x]+10

def click_event(event, x, y , flags, param):
        global draw, mask
        if event==cv2.EVENT_LBUTTONDOWN:
                draw = 1
        elif event==cv2.EVENT_RBUTTONDOWN:
                draw = 2
        elif event==cv2.EVENT_MOUSEMOVE:
                if draw == 1:
                        cv2.circle(mask,(x,y),10, (255), -1)
                elif draw == 2:
                        cv2.circle(mask,(x,y),10, (0), -1)
                cv2.imshow('mask',mask)
                cv2.waitKey(1)
        
        elif event== cv2.EVENT_LBUTTONUP:
                draw = 0
        elif event==cv2.EVENT_RBUTTONUP:
                draw = 0

cv2.namedWindow('Tracking')
cv2.createTrackbar('LH','Tracking',0,255,nothing)
cv2.createTrackbar('LS','Tracking',0,255,nothing)
cv2.createTrackbar('LV','Tracking',0,255,nothing)
cv2.createTrackbar('UH','Tracking',255,255,nothing)
cv2.createTrackbar('US','Tracking',255,255,nothing)
cv2.createTrackbar('UV','Tracking',255,255,nothing)

img = cv2.imread("room3.jpeg")
img = cv2.GaussianBlur(img,(5,5),0)
print("img",img.shape)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("img",img)
cv2.setMouseCallback('img', click_event1)
print("Select wall")
################ select wall ###########
cv2.waitKey(0)

cv2.setTrackbarPos("LH",'Tracking',lb[0])
cv2.setTrackbarPos("LS",'Tracking',lb[1])
cv2.setTrackbarPos("LV",'Tracking',lb[2])
cv2.setTrackbarPos("UH",'Tracking',ub[0])
cv2.setTrackbarPos("US",'Tracking',ub[1])
cv2.setTrackbarPos("UV",'Tracking',ub[2])
print("Move the meters")
while True:

        l_h=cv2.getTrackbarPos('LH','Tracking')
        l_s=cv2.getTrackbarPos('LS','Tracking')
        l_v=cv2.getTrackbarPos('LV','Tracking')
        u_h=cv2.getTrackbarPos('UH','Tracking')
        u_s=cv2.getTrackbarPos('US','Tracking')
        u_v=cv2.getTrackbarPos('UV','Tracking')

        lb=np.array([l_h, l_s, l_v])
        ub = np.array([u_h,u_s,u_v])

        mask = cv2.inRange(hsv, lb, ub)
        cv2.imshow("mask",resize(mask))
        cv2.imshow("img", resize(img))
        key = cv2.waitKey(1)
        if key==113:                
                break

print("lb and ub",lb, ub)
mask = cv2.inRange(hsv, lb, ub)

red = np.zeros(img.shape, np.uint8)
red[:,:,1] = 255 # replace me with any color
cv2.setMouseCallback('mask', click_event)
print("Sorry, I can't get your wall. Pait your mask")
####### mask breakpoint ##########
cv2.waitKey(0)

red = cv2.bitwise_and(red, red, mask=mask)
mask_i = cv2.bitwise_not(mask)
img = cv2.bitwise_and(img,img,mask=mask_i)
res = red+img
cv2.imshow("result",resize(res))
print("Thank you !!!")
cv2.waitKey(0)
cv2.destroyAllWindows()