#!/usr/bin/python3
import cv2

# load the overlay image. size should be smaller than video frame size
img = cv2.imread(
    'C:\\Users\\LeeSooHwan\\Desktop\\interactiveZoom\\questionSmall.png')

# # Save the transparency channel alpha
# *_, alpha = cv2.split(img)

# img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get Image dimensions
img_height, img_width, _ = img.shape

# Start Capture
cap = cv2.VideoCapture(0)

# Get frame dimensions
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Print dimensions
print('image dimensions (HxW):', img_height, "x", img_width)
print('frame dimensions (HxW):', int(frame_height), "x", int(frame_width))

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

    # add image to frame
    frame[y:y+img_height, x:x+img_width] = img

    # Display the resulting frame
    cv2.imshow('Webcam & PNG Test', frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(20) & 0xFF == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
