import pytesseract
from datasets import load_dataset
from pathlib import Path
from PIL import image


class TesseractModel:
    def __init__(self):
        pass

    def process_ocr(self, image, image_path):
        image_path = Path(image_path)
        text = pytesseract.image_to_string(image)
        data = {
            "image_path": str(image_path),
            "ocr_engine": "Tesseract OCR",
            "ocr_result": []
        }
        boxes = pytesseract.image_to_boxes(image)

    def get_boxes(image):
        output_boxes = pytesseract.image_to_boxes(image)
        result = output_boxes.split('\n')
        boxes = []
        for res in result:
            line = res.split(' ')
            if len(line) > 1 and line[0] != '~':
                box = []
                box.append(int(line[1]))
                box.append(int(line[2]))
                box.append(int(line[3]))
                box.append(int(line[4]))
                #box.append(line[5])
                boxes.append[box]

        return boxes

    def get_text(image):
        output_text = pytesseract.image_to_string(image)
        result_text = output_text.split('\n')
        characters = []
        for line in result_text:
            line = line.replace(" ", "")
            line_characters = list(line)
            characters.extend(line_characters)        