# import cv2
# from PIL import Image

# image = Image.open("questionMark.png")
# image.convert("RGBA")
# canvas = Image.new('RGBA', image.size, (255, 255, 255, 255)
#                    )  # Empty canvas colour (r,g,b,a)
# # Paste the image onto the canvas, using it's alpha channel as mask
# canvas.paste(image, mask=image)
# canvas.save("quesitonMARK.png", format="PNG")

import cv2

background = cv2.imread('question.png')
overlay = cv2.imread('questionSmall.png', cv2.IMREAD_UNCHANGED)

print(overlay.shape)  # must be (x,y,4)
print(background.shape)  # must be (x,y,3)

# downscale logo by half and position on bottom right reference
out = logoOverlay(background, overlay, scale=0.5, y=-100, x=-100)

cv2.imshow("test", out)
cv2.waitKey(0)
