from PIL import Image, ImageDraw, ImageFont
import os
import random
import csv

# Define the directory to save the synthetic data
output_dir = "synthetic_data_new"
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

# Define the number of "Hello, World!" instances in each image
min_instances = 1
max_instances = 10

# Function to check if two rectangles intersect
def intersect(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + h2)

# Function to generate synthetic data for a single image
def generate_synthetic_image(font_path, image_width, image_height):
    # Load the font
    try:
        font_size = random.randint(min_font_size, max_font_size)
        font = ImageFont.truetype(font_path, font_size)
        
        # Create a new image with a white background
        image = Image.new("RGB", (image_width, image_height), "white")
        
        # Get a drawing context
        draw = ImageDraw.Draw(image)
        
        # List to store bounding box coordinates
        bounding_boxes = []
        
        # Randomly select the number of "Hello, World!" instances
        num_instances = random.randint(min_instances, max_instances)
        for _ in range(num_instances):
            max_attempts = 100
            for _ in range(max_attempts):
                # Calculate text size and position
                text_width, text_height = draw.textsize(text, font=font)
                x = random.randint(0, image_width - text_width)
                y = random.randint(0, image_height - text_height)
                new_box = (x, y, text_width, text_height)
                
                # Check if the new bounding box intersects with existing bounding boxes
                intersect_flag = False
                for bbox in bounding_boxes:
                    if intersect(bbox, new_box):
                        intersect_flag = True
                        break
                
                if not intersect_flag:
                    # No intersection found, render the text on the image
                    draw.text((x, y), text, fill="black", font=font)
                    
                    # Append the bounding box coordinates
                    bounding_boxes.append(new_box)
                    break
            else:
                print("Warning: Failed to find non-overlapping position after maximum attempts.")
        
        # Save the image
        image_path = os.path.join(output_dir, f"synthetic_image_{os.path.splitext(os.path.basename(font_path))[0]}_{image_width}x{image_height}.png")
        image.save(image_path)
        
        return image_path, num_instances, bounding_boxes
    except:
        return None, None, None
# Generate synthetic data for each font
dataset = []
for font_file in fonts:
    font_path = os.path.join("fonts", font_file)
    
    # Generate synthetic images
    for _ in range(100):
        image_width = random.randint(min_image_width, max_image_width)
        image_height = random.randint(min_image_height, max_image_height)
        
        image_path, num_instances, bounding_boxes = generate_synthetic_image(font_path, image_width, image_height)
        
        # Add data to the dataset
        if image_path != None and num_instances != None and bounding_boxes != None:
            dataset.append({
                "image_path": image_path,
                "num_instances": num_instances,
                "bounding_boxes": bounding_boxes
            })

# Save dataset to CSV file
csv_file = "synthetic_dataset.csv"
with open(csv_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["image_path", "num_instances", "bounding_boxes"])
    writer.writeheader()
    writer.writerows(dataset)

print("Synthetic data generation completed.")
