import cv2, os, pandas as pd, subprocess, shutil
from datetime import datetime
from my_utils import model, draw_text, extract_plate_text

def process_video(video_path, csv_path, process_type="video", confidence_threshold=0.6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    extracted_data = []
    input_filename = os.path.splitext(os.path.basename(video_path))[0]
    input_ext = os.path.splitext(video_path)[1].lower()
    output_dir = "full_car_with_ocr"
    os.makedirs(output_dir, exist_ok=True)
    temp_frames_dir = "temp_frames"
    os.makedirs(temp_frames_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = model(frame)
        detections = results.xyxy[0]
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if conf < confidence_threshold:
                continue
            plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            plate_text = extract_plate_text(plate_crop)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            extracted_data.append([process_type, video_path, frame_count, plate_text, timestamp])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            draw_text(frame, plate_text, font_scale=2, pos=(int(x1), int(y1)))
        frame_filename = os.path.join(temp_frames_dir, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)

    cap.release()
    cv2.destroyAllWindows()

    if extracted_data:
        df = pd.DataFrame(extracted_data, columns=["Process Type", "Input Source", "Frame Number", "Number Plate", "Timestamp"])
        df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))

    output_video_path = os.path.join(output_dir, f"processed_{input_filename}{input_ext}")
    ffmpeg_command = f"ffmpeg -framerate {fps} -i {temp_frames_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {output_video_path}"
    subprocess.run(ffmpeg_command, shell=True)
    shutil.rmtree(temp_frames_dir, ignore_errors=True)
    print(f"Final processed video saved at: {output_video_path}")
