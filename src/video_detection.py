
from pathlib import Path
import argparse
import csv
import time
import winsound

import cv2
from ultralytics import YOLO


# Project paths
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "results"
    / "weapon_baseline_10"
    / "weights"
    / "best.pt"
)

LOG_PATH = ROOT / "results" / "detection_log.csv"

# Detection settings
CONFIDENCE_THRESHOLD = 0.50
IMAGE_SIZE = 320
WARNING_COOLDOWN = 5


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11 Video Detection Prototype"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to a prerecorded video file"
    )

    args = parser.parse_args()

    # Validate model and video paths
    if not MODEL_PATH.is_file():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    video_path = Path(args.source).resolve()

    if not video_path.is_file():
        print(f"ERROR: Video not found: {video_path}")
        return

    # Load the trained YOLO model
    print("Loading YOLO11 model...")
    model = YOLO(str(MODEL_PATH))

    # Open the video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Unable to open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    # Prepare detection log
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame_number = 0
    detection_count = 0
    warning_count = 0

    last_warning_time = -float("inf")

    print("Video detection started.")
    print("Press Q in the video window to quit.")

    try:
        with open(
            LOG_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as log_file:

            writer = csv.writer(log_file)

            writer.writerow([
                "frame",
                "time_seconds",
                "class",
                "confidence"
            ])

            while True:
                success, frame = cap.read()

                if not success:
                    break

                frame_number += 1

                # Run YOLO detection
                results = model.predict(
                    frame,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=IMAGE_SIZE,
                    device="cpu",
                    verbose=False
                )

                result = results[0]

                # Draw bounding boxes
                annotated_frame = result.plot()

                # Display detection status
                if len(result.boxes) > 0:
                    status = "DETECTION - REVIEW REQUIRED"
                    color = (0, 0, 255)

                    current_time = time.monotonic()

                    # Limit notification sound frequency
                    if (
                        current_time - last_warning_time
                        >= WARNING_COOLDOWN
                    ):
                        winsound.MessageBeep(
                            winsound.MB_ICONEXCLAMATION
                        )
                        last_warning_time = current_time
                        warning_count += 1

                else:
                    status = "NO DETECTION"
                    color = (0, 180, 0)

                # Draw status text
                cv2.putText(
                    annotated_frame,
                    status,
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA
                )

                # Record detections
                video_time = cap.get(
                    cv2.CAP_PROP_POS_MSEC
                ) / 1000.0

                if video_time <= 0:
                    video_time = frame_number / fps

                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    class_name = model.names[class_id]

                    writer.writerow([
                        frame_number,
                        round(video_time, 2),
                        class_name,
                        round(confidence, 3)
                    ])

                    detection_count += 1

                # Show the video
                cv2.imshow(
                    "YOLO Detection Prototype",
                    annotated_frame
                )

                # Press Q to exit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Final summary
    print("\nVideo detection finished.")
    print(f"Frames processed: {frame_number}")
    print(f"Detections recorded: {detection_count}")
    print(f"Warning sounds triggered: {warning_count}")
    print(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
