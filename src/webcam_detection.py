
from pathlib import Path
import csv
import time
import cv2
from ultralytics import YOLO

# Project folders
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "runs" / "detect" / "results"
    / "weapon_baseline_10" / "weights" / "best.pt"
)

LOG_PATH = ROOT / "results" / "webcam_detection_log.csv"

CONFIDENCE = 0.50
IMAGE_SIZE = 320


def main():
    if not MODEL_PATH.is_file():
        print("ERROR: Trained model not found.")
        print(MODEL_PATH)
        return

    print("Loading YOLO11 model...")
    model = YOLO(str(MODEL_PATH))

    print("Opening webcam...")
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    detection_count = 0
    start_time = time.monotonic()

    print("Webcam detection started.")
    print("Press Q in the camera window to stop.")

    try:
        with open(
            LOG_PATH, "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "frame",
                "time_seconds",
                "class",
                "confidence",
            ])

            while True:
                success, frame = camera.read()

                if not success:
                    print("Camera frame unavailable.")
                    break

                frame_count += 1

                predictions = model.predict(
                    frame,
                    conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    device="cpu",
                    verbose=False,
                )

                result = predictions[0]
                annotated = result.plot()

                if len(result.boxes) > 0:
                    status = "DETECTION - REVIEW REQUIRED"
                    color = (0, 0, 255)
                else:
                    status = "NO DETECTION"
                    color = (0, 180, 0)

                cv2.putText(
                    annotated,
                    status,
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                elapsed = time.monotonic() - start_time

                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())

                    writer.writerow([
                        frame_count,
                        round(elapsed, 2),
                        model.names[class_id],
                        round(confidence, 3),
                    ])

                    detection_count += 1

                cv2.imshow(
                    "YOLO11 Live Webcam Detection",
                    annotated,
                )

                # Press Q to stop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\nWebcam detection finished.")
    print(f"Frames processed: {frame_count}")
    print(f"Box detections: {detection_count}")
    print(f"Log saved: {LOG_PATH}")


if __name__ == "__main__":
    main()
