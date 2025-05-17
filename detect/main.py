import sys
from video_service import process_video
from image_service import process_image
from stream_service import process_stream

CSV_PATH = "number_plate_results.csv"

def main():
    if len(sys.argv) < 3:
        print("Usage:\n"
              "  python main.py video <video_file_path>\n"
              "  python main.py image <image_file_path>\n"
              "  python main.py stream <rtmp_url> <ivs_channel_url>")
        sys.exit(1)
    mode = sys.argv[1].lower()
    if mode == "video":
        process_video(sys.argv[2], CSV_PATH, process_type="video")
    elif mode == "image":
        process_image(sys.argv[2], CSV_PATH, process_type="image")
    elif mode == "stream":
        if len(sys.argv) < 4:
            print("For stream mode, provide both RTMP input and IVS output URLs.")
            sys.exit(1)
        process_stream(sys.argv[2], sys.argv[3], CSV_PATH, process_type="stream")
    else:
        print("Unknown mode. Use 'video', 'image', or 'stream'.")

if __name__ == "__main__":
    main()
