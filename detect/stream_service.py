import cv2, os, pandas as pd, subprocess
from datetime import datetime
from my_utils import model, draw_text, extract_plate_text

def process_stream(rtmp_url, ivs_channel_url, csv_path, process_type="stream", confidence_threshold=0.4):
    cap = cv2.VideoCapture(rtmp_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    extracted_data = []
    ffmpeg_command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720',
        '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-f', 'flv', ivs_channel_url
    ]
    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        results = model(frame)
        detections = results.xyxy[0]
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if conf < confidence_threshold:
                continue
            plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            plate_text = extract_plate_text(plate_crop)
            extracted_data.append([process_type, rtmp_url, frame_count, plate_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            draw_text(frame, plate_text, font_scale=2, pos=(int(x1), int(y1)))
        ffmpeg_process.stdin.write(frame.tobytes())
    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    cv2.destroyAllWindows()
    if extracted_data:
        df = pd.DataFrame(extracted_data, columns=["Process Type", "Input Source", "Frame Number", "Number Plate", "Timestamp"])
        df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
