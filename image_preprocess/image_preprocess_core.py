import cv2 

class CropImage:
    def __init__(self):
        pass
    
    def convert_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
      
    def gussian_blur(self, gray_image, kernal_size, smoothening_level):
        return cv2.GaussianBlur(
            gray_image,
            kernal_size,
            smoothening_level
        )
        
        
    def thresholding(self, blurred_image, threshold, higher_intensity):
        _, binary = cv2.threshold(
            blurred_image,
            threshold,
            higher_intensity,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary
        
    def close_small_gaps(self, binary_image, kernal_size):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, kernal_size
        )

        closed_image = cv2.morphologyEx(
            binary_image,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )
        return closed_image
    
    
    def get_sorted_contour(self, closed_image):
        contours, _ = cv2.findContours(
            closed_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        sorted_contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )
        print("Number of contours:", len(sorted_contours))
        return sorted_contours
    
    
    def crop_image(self, original_image, contours):
        if not contours:
            raise ValueError("No contours found; cannot crop the image")

        largest = contours[0]
        area = cv2.contourArea(largest)
        x, y, w, h = cv2.boundingRect(largest)

        print("Area:", area)
        print("Bounding box:", x, y, w, h)

        return original_image[y:y + h, x:x + w]
    
        