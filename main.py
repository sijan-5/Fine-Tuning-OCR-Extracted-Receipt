import cv2
from pathlib import Path
from image_preprocess.image_preprocess_wrapper import ImagePreprocessorWrapper
from OCR.paddleocr import OCRModel
# eg: zenodo 200 with labels (200)/1008-receipt.jpg"

if __name__ == "__main__":
   # Folder containing receipt images
   image_folder = Path("/Users/sijanbhattarai/Receipt Error Detection/datasets/zenodo_200")
   image_preprocess_wrapper = ImagePreprocessorWrapper()
   ocr = OCRModel()
   for image_path in sorted(image_folder.glob("*.jpg")):
       print(f"Processing: {image_path.name}")
       image = cv2.imread(str(image_path))
       if image is None:
           print(f"Could not load: {image_path}")
           continue
       # Preprocess image
       cropped_image = image_preprocess_wrapper.preprocess_image(image)
       if cropped_image is None:
           print(f"Preprocessing failed: {image_path.name}")
           continue
       ocr.process_ocr(cropped_image, image_path)
       print(f"Completed: {image_path.name}")
