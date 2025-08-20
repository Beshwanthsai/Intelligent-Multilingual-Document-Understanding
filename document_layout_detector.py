import os
import cv2
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def detect_layout(image_path):
    """
    A more reliable approach to document layout detection using edge detection and contour analysis.
    This function handles various document types and is not dependent on deep learning models.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
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

def process_image(image_file, images_folder, output_folder, create_vis=False):
    """Process a single image file and return the time taken"""
    start_time = time.time()
    
    image_path = os.path.join(images_folder, image_file)
    base_name = os.path.splitext(image_file)[0]
    json_path = os.path.join(output_folder, f"{base_name}.json")
    
    # Detect layout
    results = detect_layout(image_path)
    
    # Save results to JSON file
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Create visualization if requested
    if create_vis:
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
    
    elapsed = time.time() - start_time
    return image_file, elapsed

def main():
    """Main function to process all document images"""
    images_folder = "images"
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    print(f"Found {total_images} images to process")
    
    # Process images in parallel
    total_start_time = time.time()
    
    # Control the number of worker processes to avoid overwhelming the system
    max_workers = min(os.cpu_count(), 4)  # Use at most 4 workers
    processed_count = 0
    batch_size = 100  # Process in batches to show progress
    
    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        batch = image_files[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1} ({batch_start} to {batch_end-1})...")
        batch_start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # For first few images, create visualizations
            futures = [
                executor.submit(
                    process_image, 
                    img_file, 
                    images_folder, 
                    output_folder,
                    create_vis=(processed_count < 5)  # Only create vis for first 5 images
                ) 
                for img_file in batch
            ]
            
            for future in as_completed(futures):
                try:
                    img_file, elapsed = future.result()
                    processed_count += 1
                    
                    # Print progress occasionally
                    if processed_count % 10 == 0 or processed_count == total_images:
                        percent = (processed_count / total_images) * 100
                        print(f"Processed {processed_count}/{total_images} ({percent:.1f}%)")
                except Exception as e:
                    print(f"Error processing image: {e}")
        
        batch_elapsed = time.time() - batch_start_time
        print(f"Batch processed in {batch_elapsed:.2f} seconds")
    
    total_elapsed = time.time() - total_start_time
    print(f"All {processed_count} images processed in {total_elapsed:.2f} seconds")
    print(f"Average time per image: {total_elapsed/processed_count:.4f} seconds")

if __name__ == "__main__":
    main()
