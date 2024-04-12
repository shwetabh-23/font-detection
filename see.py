from PIL import Image, ImageDraw

import cv2

def visualize_bounding_boxes(image_path, bounding_boxes):
    # Read the image
    image = cv2.imread(image_path)
    
    # Draw bounding boxes
    for box in bounding_boxes:
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Draw red rectangle
    
    # Display the image
    cv2.imshow("Image with bounding boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example bounding box coordinates
bounding_boxes = [(4, 41, 177, 33), (183, 7, 177, 33), (198, 59, 177, 33), (0, 1, 177, 33)]

# Example image path
image_path = r"synthetic_data_new/synthetic_image_Arimo-Regular_390x108.png"  # Replace with the path to your image

# Visualize bounding boxes on the image
visualize_bounding_boxes(image_path, bounding_boxes)
