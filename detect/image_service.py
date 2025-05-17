import cv2, os, pandas as pd
from datetime import datetime
from my_utils import model, draw_text, extract_plate_text

def process_image(image_path, csv_path, process_type="image"):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    results = model(img)
    detections = results.xyxy[0]
    extracted_data = []
    img_with_boxes = img.copy()
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)])
        plate_crop = img[y1:y2, x1:x2]
        plate_text = extract_plate_text(plate_crop)
        draw_text(img_with_boxes, plate_text, font_scale=2, pos=(x1, y1))
        extracted_data.append([process_type, image_path, None, plate_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    output_dir = "full_car_with_ocr"
    os.makedirs(output_dir, exist_ok=True)
    full_filename = f"processed_{os.path.splitext(os.path.basename(image_path))[0]}{os.path.splitext(image_path)[1]}"
    full_path = os.path.join(output_dir, full_filename)
    cv2.imwrite(full_path, img_with_boxes)
    if extracted_data:
        df = pd.DataFrame(extracted_data, columns=["Process Type", "Input Source", "Frame Number", "Number Plate", "Timestamp"])
        df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
    print(f"Processed: {image_path} - {len(detections)} plates detected and saved.")
