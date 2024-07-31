import numpy as np
import cv2
from segment_anything import SamPredictor, SamAutomaticMaskGenerator

def is_valid_image(masked_image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate area and variance
    area = cv2.countNonZero(gray_image)
    variance = np.var(gray_image)
    
    # Calculate colorfulness
    (B, G, R) = cv2.split(masked_image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    colorfulness = np.sqrt(np.mean(rg ** 2) + np.mean(yb ** 2))

    # Calculate aspect ratio
    height, width = gray_image.shape
    aspect_ratio = width / height if height != 0 else 0

    # Detect significant features (e.g., edges)
    edges = cv2.Canny(gray_image, 100, 200)
    edge_density = np.sum(edges) / (height * width)

    # Texture analysis (using Local Binary Pattern)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= hist.sum()  # Normalize the histogram
    texture_score = np.sum(hist)

    # Heuristic thresholds for a valid image
    min_area = 500  # Minimum area threshold
    min_variance = 500  # Minimum variance threshold
    min_colorfulness = 15  # Minimum colorfulness threshold
    aspect_ratio_range = (0.5, 2.0)  # Aspect ratio range for typical images
    min_edge_density = 0.01  # Minimum edge density
    texture_threshold = 0.1  # Minimum texture score

    return (area > min_area and 
            variance > min_variance and 
            colorfulness > min_colorfulness and 
            aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1] and 
            edge_density > min_edge_density and 
            texture_score > texture_threshold)

def segment_images(image_bytes):
    # Load image
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Initialize SAM
    mask_generator = SamAutomaticMaskGenerator()
    
    # Generate masks
    masks = mask_generator.generate(image)
    
    # Extract images
    extracted_images = []
    for mask in masks:
        masked_image = cv2.bitwise_and(image, image, mask=mask['segmentation'].astype(np.uint8))
        
        if is_valid_image(masked_image):
            success, encoded_image = cv2.imencode('.png', masked_image)
            if success:
                extracted_images.append(encoded_image.tobytes())
    
    return extracted_images

# Required for texture analysis
def local_binary_pattern(image, P, R, method='uniform'):
    radius = R
    n_points = P
    lbp = np.zeros_like(image, dtype=np.uint8)
    for i in range(radius, image.shape[0] - radius):
        for j in range(radius, image.shape[1] - radius):
            lbp[i, j] = calculate_lbp(image, i, j, radius, n_points)
    return lbp

def calculate_lbp(image, x, y, radius, n_points):
    center = image[x, y]
    lbp_code = 0
    for point in range(n_points):
        angle = 2.0 * np.pi * point / n_points
        dx = int(radius * np.cos(angle))
        dy = int(radius * np.sin(angle))
        neighbor = image[x + dx, y + dy]
        lbp_code |= (1 << point) if neighbor > center else 0
    return lbp_code

def load_image(image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return image_bytes

def save_image(image_bytes, output_path):
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    cv2.imwrite(output_path, image)

def main():
    # Path to the input image
    image_path = '~/print.png'
    image_bytes = load_image(image_path)

    # Segment images
    segmented_images = segment_images(image_bytes)

    # Save and display segmented images
    for i, img_bytes in enumerate(segmented_images):
        output_path = f'segmented_image_{i}.png'
        save_image(img_bytes, output_path)
        print(f'Segmented image saved to {output_path}')

    print(f'Total valid segmented images: {len(segmented_images)}')

if __name__ == '__main__':
    main()
