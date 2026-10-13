
from pathlib import Path
import hashlib
import math
import subprocess
import tempfile
import time
import winsound

import cv2
import imageio_ffmpeg
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

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

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = RESULTS_DIR / "dashboard_detection_log.csv"

CONFIDENCE_THRESHOLD = 0.50
IMAGE_SIZE = 320

LOG_COLUMNS = [
    "frame",
    "time_seconds",
    "class",
    "confidence",
]


# ==========================================
# 2. STREAMLIT SETTINGS
# ==========================================

st.set_page_config(
    page_title="YOLO11 Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# ==========================================
# 3. DASHBOARD HEADER
# ==========================================

st.title("🛡️ YOLO11 Detection Dashboard")

st.subheader(
    "Weapon Detection and Early Warning System"
)

st.caption(
    "YOLO11 | Python | OpenCV | Streamlit"
)

st.warning(
    "Educational prototype only. "
    "AI predictions may be incorrect and require "
    "human verification. This is not an emergency "
    "response or certified surveillance system."
)


# ==========================================
# 4. SYSTEM OVERVIEW
# ==========================================

st.header("System Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("AI Model", "YOLO11n")

with col2:
    st.metric(
        "Model Status",
        "Ready" if MODEL_PATH.is_file() else "Not Found",
    )

with col3:
    st.metric("Test mAP@50", "67.5%")


# ==========================================
# 5. PERFORMANCE CHART
# ==========================================

st.divider()

st.header("Object Detection Performance")

performance = pd.DataFrame(
    {
        "Object": [
            "Handgun",
            "Knife",
            "Rifle",
            "Sword",
        ],
        "mAP@50 (%)": [
            82.2,
            54.6,
            64.8,
            68.5,
        ],
    }
)

st.bar_chart(performance.set_index("Object"))

st.caption(
    "Evaluation results from 186 independent test "
    "images. These results do not guarantee "
    "accurate performance on new footage."
)


# ==========================================
# 6. VIDEO ANALYSIS FUNCTION
# ==========================================

def analyze_video(uploaded_file, file_bytes):

    model = load_model()
    start_time = time.monotonic()

    detection_rows = []
    detected_classes = set()

    frame_count = 0
    detection_count = 0
    frames_with_detections = 0

    cap = None
    video_writer = None

    preview = st.empty()
    status_area = st.empty()
    progress = st.progress(0)
    frame_counter = st.empty()

    with tempfile.TemporaryDirectory() as folder:

        folder = Path(folder)

        extension = (
            Path(uploaded_file.name).suffix.lower()
            or ".mp4"
        )

        input_path = folder / f"input{extension}"
        raw_path = folder / "processed_raw.mp4"
        final_path = folder / "processed_h264.mp4"

        input_path.write_bytes(file_bytes)

        try:
            cap = cv2.VideoCapture(str(input_path))

            if not cap.isOpened():
                raise RuntimeError(
                    "Unable to open the uploaded video."
                )

            fps = cap.get(cv2.CAP_PROP_FPS)

            if not math.isfinite(fps) or fps <= 0:
                fps = 30.0

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if width <= 0 or height <= 0:
                raise RuntimeError(
                    "Invalid video dimensions."
                )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            video_writer = cv2.VideoWriter(
                str(raw_path),
                fourcc,
                fps,
                (width, height),
            )

            if not video_writer.isOpened():
                raise RuntimeError(
                    "Unable to create processed video."
                )

            status_area.info(
                "YOLO11 video analysis is running..."
            )

            # Process video frame by frame
            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame_count += 1

                predictions = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=IMAGE_SIZE,
                    device="cpu",
                    verbose=False,
                )

                result = predictions[0]
                boxes = result.boxes

                # Draw YOLO bounding boxes
                annotated_frame = result.plot()

                # Save every annotated frame
                video_writer.write(annotated_frame)

                if len(boxes) > 0:
                    frames_with_detections += 1

                video_time = frame_count / fps

                # Record predictions
                for box in boxes:

                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())

                    class_name = model.names[class_id]

                    detected_classes.add(class_name)
                    detection_count += 1

                    detection_rows.append(
                        {
                            "frame": frame_count,
                            "time_seconds": round(
                                video_time, 2
                            ),
                            "class": class_name,
                            "confidence": round(
                                confidence, 3
                            ),
                        }
                    )

                # Show analysis preview
                if (
                    frame_count == 1
                    or frame_count % 10 == 0
                    or len(boxes) > 0
                ):

                    rgb_frame = cv2.cvtColor(
                        annotated_frame,
                        cv2.COLOR_BGR2RGB,
                    )

                    preview.image(
                        rgb_frame,
                        channels="RGB",
                        use_container_width=True,
                    )

                    if len(boxes) > 0:
                        status_area.warning(
                            "DETECTION - HUMAN REVIEW REQUIRED"
                        )
                    else:
                        status_area.info(
                            "No detection in current frame"
                        )

                if total_frames > 0:
                    progress.progress(
                        min(
                            frame_count / total_frames,
                            1.0,
                        )
                    )

                if frame_count % 10 == 0:
                    frame_counter.caption(
                        f"Frames processed: {frame_count}"
                    )

        finally:
            if cap is not None:
                cap.release()

            if video_writer is not None:
                video_writer.release()

        if frame_count == 0:
            raise RuntimeError(
                "Video contains no readable frames."
            )

        # ==================================
        # CONVERT VIDEO TO H.264
        # ==================================

        status_area.info(
            "Finalizing processed video..."
        )

        video_to_download = raw_path
        conversion_warning = None

        try:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

            command = [
                ffmpeg,
                "-y",
                "-i",
                str(raw_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-an",
                "-movflags",
                "+faststart",
                str(final_path),
            ]

            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )

            if (
                not final_path.is_file()
                or final_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    "H.264 conversion failed."
                )

            video_to_download = final_path

        except Exception as error:
            conversion_warning = (
                "H.264 conversion failed. "
                "The MP4 will still be available "
                "for download, but browser playback "
                f"may not work. Details: {error}"
            )

        # ==================================
        # SAVE CSV DETECTION HISTORY
        # ==================================

        history = pd.DataFrame(
            detection_rows,
            columns=LOG_COLUMNS,
        )

        history.to_csv(
            LOG_PATH,
            index=False,
        )

        video_bytes = video_to_download.read_bytes()
        elapsed = time.monotonic() - start_time

        progress.progress(1.0)

        if detection_count > 0:
            status_area.warning(
                "Analysis complete: predictions "
                "require human review."
            )
        else:
            status_area.info(
                "Analysis complete: no predictions "
                "above the configured threshold."
            )

        return {
            "video_bytes": video_bytes,
            "frame_count": frame_count,
            "detection_count": detection_count,
            "frames_with_detections": (
                frames_with_detections
            ),
            "detected_classes": sorted(
                detected_classes
            ),
            "history": history,
            "elapsed": elapsed,
            "conversion_warning": conversion_warning,
        }


# ==========================================
# 7. VIDEO UPLOAD
# ==========================================

st.divider()

st.header("Video Monitoring")

uploaded_video = st.file_uploader(
    "Upload video for YOLO analysis",
    type=["mp4", "mov", "avi"],
)

if uploaded_video is not None:

    st.success("Video uploaded successfully!")

    st.subheader("Original Video")

    if uploaded_video.name.lower().endswith(".mp4"):
        st.video(uploaded_video)
    else:
        st.info(
            "MOV and AVI files may require conversion "
            "for browser playback."
        )

    file_bytes = uploaded_video.getvalue()

    upload_id = hashlib.sha256(
        file_bytes
    ).hexdigest()

    if st.button(
        "🔍 Analyze Video",
        type="primary",
    ):

        st.session_state.analysis_result = None

        if not MODEL_PATH.is_file():
            st.error("Trained YOLO model not found.")

        else:
            try:
                with st.spinner(
                    "Analyzing video with YOLO11..."
                ):

                    analysis_result = analyze_video(
                        uploaded_video,
                        file_bytes,
                    )

                    analysis_result["upload_id"] = (
                        upload_id
                    )

                    st.session_state.analysis_result = (
                        analysis_result
                    )

            except Exception as error:
                st.error(
                    f"Video analysis failed: {error}"
                )


# ==========================================
# 8. ANALYSIS RESULTS
# ==========================================

analysis = st.session_state.analysis_result

if (
    analysis is not None
    and uploaded_video is not None
    and analysis["upload_id"] == upload_id
):

    st.divider()
    st.header("Analysis Results")

    st.success("Video analysis completed!")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Frames Analyzed",
            analysis["frame_count"],
        )

    with r2:
        st.metric(
            "Box Detections",
            analysis["detection_count"],
        )

    with r3:
        st.metric(
            "Frames Flagged",
            analysis["frames_with_detections"],
        )

    st.caption(
        f"Processing time: {analysis['elapsed']:.1f} seconds"
    )

    st.caption(
        "Box detections are model predictions "
        "across frames, not unique physical objects."
    )

    # ======================================
    # STEP 70: DETECTION WARNING SYSTEM
    # ======================================

    st.subheader("Detection Warning System")

    if analysis["detection_count"] > 0:

        st.warning(
            "⚠️ OBJECT DETECTED — HUMAN REVIEW REQUIRED"
        )

        st.write(
            "**Predicted classes:** "
            + ", ".join(
                analysis["detected_classes"]
            )
        )

        st.write(
            "**Frames with detections:** "
            + str(
                analysis["frames_with_detections"]
            )
        )

        st.info(
            "These predictions have not been "
            "independently verified. Review the "
            "annotated footage before drawing "
            "conclusions."
        )

        # Plays a sound on the Windows computer
        # running the Streamlit server.
        if st.button(
            "🔔 Play Detection Alert",
            key="play_detection_alert",
        ):

            try:
                winsound.MessageBeep(
                    winsound.MB_ICONEXCLAMATION
                )

                st.success(
                    "Detection alert sound triggered "
                    "on this Windows computer."
                )

            except Exception as error:

                st.error(
                    f"Unable to play alert: {error}"
                )

    else:

        st.info(
            "No predictions were recorded above "
            "the confidence threshold. This does "
            "not guarantee that the footage is safe."
        )

    # ======================================
    # PROCESSED VIDEO
    # ======================================

    st.subheader("Processed Video")

    if analysis["conversion_warning"] is None:

        st.video(
            analysis["video_bytes"],
            format="video/mp4",
        )

    else:

        st.warning(
            analysis["conversion_warning"]
        )

    # ======================================
    # DOWNLOAD BUTTONS
    # ======================================

    st.download_button(
        label="📥 Download Processed Video",
        data=analysis["video_bytes"],
        file_name="YOLO_Processed_Video.mp4",
        mime="video/mp4",
        use_container_width=True,
    )

    st.download_button(
        label="📥 Download Detection CSV",
        data=analysis[
            "history"
        ].to_csv(index=False),
        file_name="YOLO_Detection_Log.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ==========================================
# 9. DETECTION HISTORY
# ==========================================

st.divider()

st.header("Detection History")

if LOG_PATH.is_file():

    try:
        history = pd.read_csv(LOG_PATH)

        if history.empty:

            st.info(
                "No detections recorded in "
                "the latest analysis."
            )

        else:

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True,
            )

            st.metric(
                "Recorded Box Detections",
                len(history),
            )

    except Exception as error:

        st.error(
            f"Unable to read detection log: {error}"
        )

else:

    st.info(
        "No dashboard detection history yet."
    )


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.caption(
    "YOLO11 Weapon Detection System | "
    "Information Security Project | "
    "Educational Prototype"
)
