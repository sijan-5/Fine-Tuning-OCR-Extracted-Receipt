import numpy as np
import logging
import cv2
from image_preprocess.image_preprocess_core import CropImage




class ImagePreprocessorWrapper:
   def __init__(self):
       self.crop_image = CropImage()
       pass
  
   def preprocess_image(self, original_image):
       if not isinstance(original_image, np.ndarray):
           raise TypeError("original_image must be a NumPy array")
       if original_image.size == 0:
           raise ValueError("original_image must not be empty")

       try:
           gray_image = self.crop_image.convert_gray(original_image)
           blur_image = self.crop_image.gussian_blur(gray_image, (5,5), 0)
           threshold_image = self.crop_image.thresholding(blur_image, 0, 255)
           closed_image = self.crop_image.close_small_gaps(threshold_image, (9,9))
           contours = self.crop_image.get_sorted_contour(closed_image)
      
           receipt_crop = self.crop_image.crop_image(original_image, contours)
           # cv2.imshow("Receipt Crop", receipt_crop)
           # cv2.imshow("Original Image", original_image)
           # cv2.imshow("Binary", gray_image)
           # cv2.imshow("After Closing", closed_image)
           # cv2.waitKey(0)
           # cv2.destroyAllWindows()
           return receipt_crop
  
       except cv2.error as e:
           logging.error(f"Something went wrong with CV2: {e.err}")
      
       except e:
           logging.error(f"Something went wrong with Python: {e}")
