#!/usr/bin/python3
import cv2
from PIL import Image
import numpy as np


def loadIMG(fileName):
    # load the overlay image. size should be smaller than video frame size
    # img = cv2.imread('questionSmall.png')
    img = Image.open(fileName).convert('RGBA')
    R, G, B, A = img.split()
    img = Image.merge('RGBA', (B, G, R, A))
    return img


def webcamWithPNG():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # # add image to frame
    # frame[y:y+img_height, x:x+img_width] = img

    # Make PIL image from frame, paste in speedo, revert to OpenCV frame
    pilim = Image.fromarray(frame)
    pilim.paste(img, box=(0, 20), mask=img)
    frame = np.array(pilim)
    return frame


def showWebcam(frame):
    while(True):
        # Display the resulting frame
        cv2.imshow('Webcam & PNG Test', frame)

        # Exit if ESC key is pressed
        if cv2.waitKey(20) & 0xFF == 27:
            break


if __name__ == "__main__":
    # load image through the loadIMG function
    img = loadIMG('..\\effect\\questionSmall.png')

    # Start Capture - 0 for Window, 1 for Mac
    cap = cv2.VideoCapture(1)

    # Get frame dimensions
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Start webcam with PNG
    frame = webcamWithPNG()
    showWebcam(frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
