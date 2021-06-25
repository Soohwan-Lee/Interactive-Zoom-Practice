import cv2

image = Image.open("captcha1.png")
image.convert("RGBA")
canvas = Image.new('RGBA', image.size, (255, 255, 255, 255)
                   )  # Empty canvas colour (r,g,b,a)
# Paste the image onto the canvas, using it's alpha channel as mask
canvas.paste(image, mask=image)
canvas.save("captcha1.png", format="PNG")
