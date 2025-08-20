import os
import cv2
import json
import numpy as np
import time

def detect_layout(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours on the dilated edges
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and classify contours
    results = []
    min_contour_area = 500  # Adjust this threshold as needed
    
    for contour in contours:
        # Calculate area and skip if too small
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip very small boxes
        if w < 30 or h < 20:
            continue
        
        # Simple classification based on aspect ratio and size
        aspect_ratio = float(w) / h
        
        if aspect_ratio > 5:  # Very wide rectangle
            block_type = "Text"
        elif aspect_ratio < 0.2:  # Very tall rectangle
            block_type = "Figure"
        elif w > image.shape[1] * 0.4 and h > image.shape[0] * 0.2:  # Large area
            block_type = "Table"
        elif area < 5000:  # Small area
            block_type = "Title"
        else:
            block_type = "Text"  # Default
        
        results.append({
            "bbox": [float(x), float(y), float(w), float(h)],
            "class": block_type
        })
    
    return results

def process_batch(image_files, start_idx, batch_size):
    end_idx = min(start_idx + batch_size, len(image_files))
    batch = image_files[start_idx:end_idx]
    
    print(f"Processing batch {start_idx // batch_size + 1} ({start_idx} to {end_idx-1})...")
    
    for i, image_file in enumerate(batch):
        image_path = os.path.join(images_folder, image_file)
        
        # Extract base name without extension
        base_name = os.path.splitext(image_file)[0]
        json_path = os.path.join(output_folder, f"{base_name}.json")
        
        # Detect layout
        results = detect_layout(image_path)
        
        # Save results to JSON file
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        
        # Optional: Create visualization for first few images
        if i < 5:
            image = cv2.imread(image_path)
            if image is not None:
                # Make a copy for visualization
                vis_image = image.copy()
                
                for item in results:
                    x, y, w, h = [int(v) for v in item["bbox"]]
                    class_name = item["class"]
                    
                    # Draw bounding box with different colors
                    if class_name == "Text":
                        color = (0, 255, 0)  # Green
                    elif class_name == "Title":
                        color = (255, 0, 0)  # Blue
                    elif class_name == "Table":
                        color = (0, 0, 255)  # Red
                    else:
                        color = (255, 255, 0)  # Yellow
                        
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label
                    cv2.putText(vis_image, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Save visualization
                vis_path = os.path.join(output_folder, f"{base_name}_vis.jpg")
                cv2.imwrite(vis_path, vis_image)
    
    return end_idx

# Define folders
images_folder = "images"
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
total_images = len(image_files)
print(f"Found {total_images} images to process")

# Process in batches to avoid memory issues
batch_size = 10
start_idx = 0

# Process first few batches to demonstrate
num_batches_to_process = 3
for _ in range(num_batches_to_process):
    if start_idx >= total_images:
        break
    start_time = time.time()
    start_idx = process_batch(image_files, start_idx, batch_size)
    elapsed = time.time() - start_time
    print(f"Batch processed in {elapsed:.2f} seconds")

print(f"Processed {start_idx} out of {total_images} images")
print("To process all images, adjust num_batches_to_process")
