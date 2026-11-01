import cv2
import glob
import os
from pathlib import Path
from image_preprocess.image_preprocess_wrapper import ImagePreprocessorWrapper
from OCR.paddleocr import OCRModel
from OCR.tesseractocr import TesseractModel
import pytesseract
# eg: zenodo 200 with labels (200)/1008-receipt.jpg"

def run_tesseract(dataset="SROIE"):
    if dataset == "SROIE":
        base_path_SROIE = "datasets/SROIE2019/"
        image_folder = os.path.join(base_path_SROIE, "test/chatgpt_images")
    else:
        base_path_Zenodo = "datasets/zenodo_200"
        image_folder = os.path.join(base_path_Zenodo, "Images")

    image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    image_preprocess_wrapper = ImagePreprocessorWrapper()
    ocr = TesseractModel()
    for image_path in image_files:
        print(f"Processing: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load: {image_path}")
            continue
        # Preprocess image
        cropped_image = image_preprocess_wrapper.preprocess_image(image)
        if cropped_image is None:
            print(f"Preprocessing failed: {image_path.name}")
            continue
        ocr.process_ocr(image_path, image_path)
        print(f"Completed: {image_path}")

if __name__ == "__main__":
   run_tesseract()
   # Folder containing receipt images
   #image_folder = Path("/Users/sijanbhattarai/Receipt Error Detection/datasets/zenodo_200")
   """image_folder = Path("datasets/SROIE2019/test/chatgpt_images")
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
       print(f"Completed: {image_path.name}")"""