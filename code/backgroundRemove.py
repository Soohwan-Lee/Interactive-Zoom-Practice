import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
# cap.set(cv2.CAP_PROP_FPS, 60)

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

while True:
    success, img = cap.read()
    # imgOut = segmentor.removeBG(img, (255,0,255), threshold=0.83)
    imgOut = segmentor.removeBG(img, (255, 255, 255), threshold=0.95)
    _, imgOut = fpsReader.update(imgOut)

    cv2.imshow("Background Remove Example", imgOut)
    cv2.waitKey(1)
