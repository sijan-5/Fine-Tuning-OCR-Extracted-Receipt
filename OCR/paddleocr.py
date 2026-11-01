from paddleocr import PaddleOCR
import json
from pathlib import Path


# Setting   Set True when   Set False when
# use_doc_orientation_classify  The entire receipt might be rotated by 90°, 180° or 270°.   Receipts are already upright.
# use_doc_unwarping The receipt is distorted, curved or warped by the camera perspective.   Receipts are flat and have little distortion.
# use_textline_orientation  Individual text lines might be rotated or oriented differently. Text lines are already aligned properly.
class OCRModel:
    def __init__(self):
        self.ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False
        )

    def process_ocr(self, image, image_path):
        image_path = Path(image_path)
        results = self.ocr.predict(image)
        if results:
            result = results[0]
            texts = result["rec_texts"]
            scores = result["rec_scores"]
            boxes = result["rec_polys"]
            data = {
                "image_path": str(image_path),
                "ocr_engine": "PaddleOCR",
                "ocr_result": []
            }
            for text, score, box in zip(texts, scores, boxes):
                ocr_item = {
                    "text": text,
                    "confidence": float(score),
                    "bounding_box": box.tolist()
                }
                data["ocr_result"].append(ocr_item)
            # File paths
            #text_file_path = image_path.with_suffix(".txt")
            #json_file_path = image_path.parent / f"{image_path.stem}_ocr.json"
            # File paths
            text_file_path = image_path.parent.parent / f"chatgpt_images/paddle_text_files/{image_path.stem}.txt"
            json_file_path = image_path.parent.parent / f"chatgpt_images/paddle_json_files/{image_path.stem}_ocr.json"
            # Save OCR text
            text_file_path.write_text(
                "\n".join(texts),
                encoding="utf-8"
            )
            # Save OCR JSON
            with open(json_file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
            print(f"Text saved: {text_file_path}")
            print(f"JSON saved: {json_file_path}")
