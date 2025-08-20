import os
import cv2
import json
import numpy as np
from PIL import Image

# A simple rule-based layout detection function
def detect_layout(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and classify contours
    results = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out very small contours
        if w < 50 or h < 20:
            continue
        
        # Simple classification based on aspect ratio and size
        if w > h * 3:  # Wide rectangle
            block_type = "Text"
        elif h > w * 3:  # Tall rectangle
            block_type = "Figure"
        elif w > image.shape[1] * 0.5 and h > image.shape[0] * 0.3:  # Large area
            block_type = "Table"
        elif w < 100 and h < 50:  # Small area
            block_type = "Title"
        else:
            block_type = "Text"  # Default
        
        results.append({
            "bbox": [float(x), float(y), float(w), float(h)],
            "class": block_type
        })
    
    return results

images_folder = "images"
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# Loop through all images in the images folder
for image_file in os.listdir(images_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing {image_file}...")
        image_path = os.path.join(images_folder, image_file)
        
        # Detect layout
        results = detect_layout(image_path)
        
        # Save results to JSON file
        json_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved results to {json_path}")
