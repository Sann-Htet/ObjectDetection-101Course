from PIL import Image

# Open the image file
image_path = "Project-1_CarCounter/mask.png"
original_image = Image.open(image_path)

# Resize the image to 1280x720
resized_image = original_image.resize((1280, 720))

# Save the resized image
resized_image.save("mask_resize.png")
