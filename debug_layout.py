import os
import cv2
import json
import numpy as np

def detect_layout(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return []
    
    print(f"Image loaded successfully with shape: {image.shape}")
    
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
    print(f"Found {len(contours)} contours")
    
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
    
    print(f"After filtering, found {len(results)} layout elements")
    return results

# Test on a single image
image_path = os.path.join("images", "doc_00000.png")
output_path = os.path.join("outputs", "doc_00000_debug.json")

# Detect layout
results = detect_layout(image_path)

# Save results to JSON file
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"Saved results to {output_path}")

# Create a visualization
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
    vis_path = os.path.join("outputs", "doc_00000_vis.jpg")
    cv2.imwrite(vis_path, vis_image)
    print(f"Saved visualization to {vis_path}")
    
    # Also create a test file for comparison
    test_json_path = os.path.join("outputs", "doc_00000_test.json")
    with open(test_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved test JSON to {test_json_path}")
else:
    print("Failed to load image for visualization")
