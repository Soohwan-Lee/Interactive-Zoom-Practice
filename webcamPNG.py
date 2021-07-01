#!/usr/bin/python3
import cv2
from PIL import Image
import numpy as np
import random

# load the overlay image. size should be smaller than video frame size
# img = cv2.imread('questionSmall.png')
img = Image.open('questionSmall.png').convert('RGBA')
R, G, B, A = img.split()
img = Image.merge('RGBA', (B, G, R, A))

<<<<<<< HEAD
<<<<<<< HEAD
# # Get Image dimensions
# img_height, img_width, _ = img.shape
=======
=======
>>>>>>> 8c396e549ef473f9ebe64ea542138ed9f228b5e9
# png = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


# # Save the transparency channel alpha
# *_, alpha = cv2.split(img)

# img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get Image dimensions
img_height, img_width, _ = img.shape
>>>>>>> 5aa44fd416358ca141c7ebafa0f5a402ee3f95ad

# Start Capture
cap = cv2.VideoCapture(0)

# Get frame dimensions
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# # Print dimensions
# print('image dimensions (HxW):', img_height, "x", img_width)
# print('frame dimensions (HxW):', int(frame_height), "x", int(frame_width))

# Decide X,Y location of overlay image inside video frame.
# following should be valid:
#   * image dimensions must be smaller than frame dimensions
#   * x+img_width <= frame_width
#   * y+img_height <= frame_height
# otherwise you can resize image as part of your code if required

x = 0
y = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

<<<<<<< HEAD
<<<<<<< HEAD
    # # add image to frame
    # frame[y:y+img_height, x:x+img_width] = img

    # Make PIL image from frame, paste in speedo, revert to OpenCV frame
    pilim = Image.fromarray(frame)
    pilim.paste(img, box=(0, 20), mask=img)
    frame = np.array(pilim)
=======
    # add image to frame
    frame[y:y+img_height, x:x+img_width] = img
    # added_image = cv2.addWeighted(frame, 0.4, png, 0.1, 0)
>>>>>>> 5aa44fd416358ca141c7ebafa0f5a402ee3f95ad
=======
    # add image to frame
    frame[y:y+img_height, x:x+img_width] = img
    # added_image = cv2.addWeighted(frame, 0.4, png, 0.1, 0)
>>>>>>> 8c396e549ef473f9ebe64ea542138ed9f228b5e9

    # Display the resulting frame
    cv2.imshow('Webcam & PNG Test', frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(20) & 0xFF == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
