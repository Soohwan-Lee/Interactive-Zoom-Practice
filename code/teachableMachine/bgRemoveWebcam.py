# Background Remove from Webcam and Send it to Virtual Cam
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import pyvirtualcam
from pyvirtualcam import PixelFormat

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# cap.set(cv2.CAP_PROP_FPS, 60)

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

# while True:
#     success, img = cap.read()
#     # imgOut = segmentor.removeBG(img, (255,0,255), threshold=0.83)
#     imgOut = segmentor.removeBG(img, (255, 255, 255), threshold=0.2)
#     _, imgOut = fpsReader.update(imgOut)

#     cv2.imshow("Background Remove Example", imgOut)
#     cv2.waitKey(1)


############ Try code ##############
# Query final capture device values (may be different from preferred settings).
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)

with pyvirtualcam.Camera(width, height, fps_in, fmt=PixelFormat.BGR, print_fps=fps_in) as cam:
    print(
        f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')

    while True:
        success, img = cap.read()
        # imgOut = segmentor.removeBG(img, (255,0,255), threshold=0.83)
        imgOut = segmentor.removeBG(img, (255, 255, 255), threshold=0.4)
        #_, imgOut = fpsReader.update(imgOut)

        cv2.imshow("Background Remove Example", imgOut)
        cv2.waitKey(1)

        # Send webcam video through virtual cam
        cam.send(imgOut)
        # Wait until it's time for the next frame.
        cam.sleep_until_next_frame()
