from PIL import Image, ImageDraw, ImageFont
import os
import random

# Define the directory to save the synthetic data
output_dir = "synthetic_data"
os.makedirs(output_dir, exist_ok=True)

# Define the text to render
text = "Hello, World!"

# Define the list of fonts to use
fonts = [
    "Oswald-Regular.ttf",
    "Roboto-Regular.ttf",
    "OpenSans-Regular.ttf",
    "Ubuntu-Regular.ttf",
    "PTSerif-Regular.ttf",
    "DancingScript-Regular.ttf",
    "Arimo-Regular.ttf",
    "NotoSans-Regular.ttf",
    "PatuaOne-Regular.ttf"
]

# Define the range of font sizes
min_font_size = 24
max_font_size = 48

# Define the range of image sizes
min_image_width = 300
max_image_width = 500
min_image_height = 80
max_image_height = 120

# Loop through each font
for font_file in fonts:
    # Load the font
    for _ in range(1000):

        font_path = os.path.join("fonts", font_file)
        
        # Generate random font size
        font_size = random.randint(min_font_size, max_font_size)
        
        # Load the font
        font = ImageFont.truetype(font_path, font_size)
        
        # Generate random image size
        image_width = random.randint(min_image_width, max_image_width)
        image_height = random.randint(min_image_height, max_image_height)
        
        # Create a new image with a white background
        image = Image.new("RGB", (image_width, image_height), "white")
        
        # Get a drawing context
        draw = ImageDraw.Draw(image)
        
        # Calculate text size and position
        text_width, text_height = draw.textsize(text, font=font)
        x = random.randint(0, image_width - text_width)
        y = random.randint(0, image_height - text_height)
        
        # Render the text on the image
        draw.text((x, y), text, fill="black", font=font)
        
        # Save the image
        output_path = os.path.join(output_dir, f"synthetic_{os.path.splitext(font_file)[0]}_{font_size}_{image_width}x{image_height}.png")
        image.save(output_path)
    
print("Synthetic data generation completed.")
