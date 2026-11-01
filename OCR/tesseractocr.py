import cv2
import json
import pandas as pd
import pytesseract
from datasets import load_dataset
from pathlib import Path
from PIL import Image
from pytesseract import Output

class TesseractModel:
    def __init__(self):
        pass

    def process_ocr(self, image, image_path):
        image_path = Path(image_path)
        results = pytesseract.image_to_data(image, config="--psm 3", output_type=Output.DATAFRAME)\

        results = results[
            (results["conf"] >= 0) &
            (results["text"].notna()) &
            (results["text"].astype(str).str.strip() != "")
        ]

        # Group words into lines
        line_groups = results.groupby(
            ["page_num", "block_num", "par_num", "line_num"],
            sort=False
        )

        data = {
            "image_path": str(image_path),
            "ocr_engine": "Tesseract OCR",
            "ocr_result": []
        }

        texts = []
        for _, group in line_groups:
            # Combine words in reading order
            text = " ".join(group["text"].astype(str).str.strip())

            # Calcualte average word confidence, normalized to 0 - 1
            confidence = float(group["conf"].mean() / 100.0)

            # Bounding box enclosing the entire line: x1, y1, x2, y2
            x1 = int(group["left"].min())
            y1 = int(group["top"].min())
            x2 = int((group["left"] + group["width"]).max())
            y2 = int((group["top"] + group["height"]).max())

            box = [x1, y1, x2, y2]

            ocr_item = {
                "text": text,
                "confidence": confidence,
                "bounding_box": box
            }

            data["ocr_result"].append(ocr_item)

            texts.append(text)
        # File paths
        text_file_path = image_path.parent / f"text_files/{image_path.stem}.txt"
        json_file_path = image_path.parent / f"json_files/{image_path.stem}_ocr.json"
        
        #texts = pytesseract.image_to_string(image, config="--psm 3")
        #print(texts)
        # Save OCR text
        text_file_path.write_text(
            "\n".join(texts),
            encoding="utf-8-sig"
        )
        # Writing to a new file or overwriting an existing file
        """with open(text_file_path, 'w', encoding="utf-8-sig") as f:
            f.write(texts)"""

        # Save OCR JSON
        with open(json_file_path, "w", encoding="utf-8-sig") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

        print(f"Text saved: {text_file_path}")
        print(f"JSON saved: {json_file_path}")
        