import torch
import cv2
from paddleocr import PaddleOCR

# Load model and OCR once
model = torch.hub.load('ultralytics/yolov5', 'custom',
                      path='/video_enhancement/Automatic-Number-Plate-Recognition-using-YOLOv5/Weights/best.pt',
                      force_reload=True)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def draw_text(img, text, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), font_scale=3, font_thickness=2,
              text_color=(0, 255, 0), text_color_bg=(0, 0, 0)):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return text_size

def extract_plate_text(plate_crop):
    try:
        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        ocr_results = ocr.ocr(gray_plate, cls=True)
        return " ".join([word[1][0] for result in ocr_results for word in result]) if ocr_results else "N/A"
    except Exception:
        return "N/A"
