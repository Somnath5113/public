# Number Plate Detection & OCR

This project provides tools for automatic number plate detection and recognition from videos, images, and live streams using YOLOv5 and PaddleOCR.

## Features

- Detects number plates in images, videos, and RTMP streams
- Extracts plate text using OCR
- Annotates frames with detected plate numbers
- Saves results and processed media
- Outputs results to a CSV file

## Requirements

- Python 3.8+
- [YOLOv5](https://github.com/ultralytics/yolov5) (via torch.hub)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- OpenCV
- pandas
- torch


## Installation

Clone the repository
```sh
cd detect/
```

Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage

Run the main script with one of the following modes:

### Video File
```sh
python main.py video <video_file_path>
```

### Image File
```sh
python main.py image <image_file_path>
```

### RTMP Stream
```sh
python main.py stream <rtmp_url> <ivs_channel_url>
```

- Processed videos/images are saved in the `full_car_with_ocr` directory.
- Results are appended to `number_plate_results.csv`.

## File Structure

- [`main.py`](main.py): Entry point for running detection.
- [`video_service.py`](video_service.py): Video processing logic.
- [`image_service.py`](image_service.py): Image processing logic.
- [`stream_service.py`](stream_service.py): RTMP stream processing logic.
- [`my_utils.py`](my_utils.py): Shared model loading, OCR, and drawing utilities.

## Customization

- Update the YOLOv5 weights path in [`my_utils.py`](my_utils.py) as needed.
- Adjust confidence thresholds in the service scripts for your use case.

## License

