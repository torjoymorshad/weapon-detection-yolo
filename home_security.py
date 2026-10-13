
import os

# Reduce memory usage on an 8 GB RAM laptop.
# These settings must be applied before importing NumPy or PyTorch.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import base64
import csv
import time
from datetime import datetime, time as dtime
from email.message import EmailMessage
from pathlib import Path

import cv2
import streamlit as st
import torch
from ultralytics import YOLO


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

WEAPON_WEIGHTS = ROOT / "models" / "weapon_best.pt"
LOG = ROOT / "results" / "home_security_log.csv"
LOG.parent.mkdir(parents=True, exist_ok=True)

GMAIL_RECIPIENT = os.getenv("GMAIL_RECIPIENT", "")

torch.set_num_threads(2)

st.set_page_config(
    page_title="YOLO11 Smart Home Security",
    layout="wide",
)

st.title("YOLO11 Smart Home Security Monitoring")

st.caption(
    "Academic research prototype. "
    "AI detections may be incorrect and require human verification."
)


# ============================================================
# LOAD PERSON AND WEAPON MODELS
# ============================================================

@st.cache_resource
def load_models():
    person_model = YOLO("yolo11n.pt")

    weapon_model = None

    if WEAPON_WEIGHTS.exists():
        weapon_model = YOLO(str(WEAPON_WEIGHTS))

    return person_model, weapon_model


# ============================================================
# MONITORING SCHEDULE
# ============================================================

def within_hours(start, end, now):
    if start == end:
        return True

    if start < end:
        return start <= now < end

    return now >= start or now < end


# ============================================================
# GMAIL NOTIFICATION
# ============================================================

def send_email(subject, body):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    token_file = ROOT / "token.json"

    if not token_file.exists():
        return "ERROR: token.json not found."

    scopes = [
        "https://www.googleapis.com/auth/gmail.send"
    ]

    try:
        credentials = Credentials.from_authorized_user_file(
            str(token_file),
            scopes,
        )

        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())

            token_file.write_text(
                credentials.to_json(),
                encoding="utf-8",
            )

        if not credentials.valid:
            return "ERROR: Gmail authorization expired."

        message = EmailMessage()
        message["To"] = GMAIL_RECIPIENT
        message["Subject"] = subject
        message.set_content(body)

        encoded = base64.urlsafe_b64encode(
            message.as_bytes()
        ).decode("utf-8")

        service = build(
            "gmail",
            "v1",
            credentials=credentials,
            cache_discovery=False,
        )

        service.users().messages().send(
            userId="me",
            body={"raw": encoded},
        ).execute()

        return "Gmail alert sent successfully."

    except Exception as error:
        return f"Gmail error: {error}"


# ============================================================
# SAVE DETECTION HISTORY
# ============================================================

def log_event(event, confidence, source):
    file_exists = LOG.exists()

    with LOG.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "event",
                "confidence",
                "source",
            ])

        writer.writerow([
            datetime.now().isoformat(
                timespec="seconds"
            ),
            event,
            round(confidence, 3),
            source,
        ])


# ============================================================
# DRAW DETECTION BOXES
# ============================================================

def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0].tolist(),
    )

    confidence = float(box.conf[0])

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    cv2.putText(
        frame,
        f"{label} {confidence:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:
    st.header("Camera and Security Settings")

    source_type = st.selectbox(
        "Camera source",
        [
            "Laptop webcam",
            "IP camera",
        ],
    )

    camera_index = 0
    stream_url = ""

    if source_type == "Laptop webcam":
        camera_index = st.number_input(
            "Webcam number",
            min_value=0,
            max_value=10,
            value=0,
        )

    else:
        stream_url = st.text_input(
            "IP camera stream URL",
            type="password",
        )

    start_hour = st.time_input(
        "Monitoring starts",
        value=dtime(23, 0),
    )

    end_hour = st.time_input(
        "Monitoring ends",
        value=dtime(5, 0),
    )

    person_threshold = st.slider(
        "Person confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )

    weapon_threshold = st.slider(
        "Weapon confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.20,
        step=0.05,
    )

    person_cooldown = st.slider(
        "Person alert cooldown (seconds)",
        min_value=10,
        max_value=600,
        value=60,
        step=10,
    )

    weapon_cooldown = st.slider(
        "Weapon alert cooldown (seconds)",
        min_value=10,
        max_value=600,
        value=30,
        step=10,
    )

    email_enabled = st.checkbox(
        "Send Gmail security alerts",
        value=False,
    )

    st.info(
        "This prototype does not identify individual "
        "people or distinguish family members from visitors."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "capture" not in st.session_state:
    st.session_state.capture = None

if "running" not in st.session_state:
    st.session_state.running = False

if "last_person_alert" not in st.session_state:
    st.session_state.last_person_alert = 0.0

if "last_weapon_alert" not in st.session_state:
    st.session_state.last_weapon_alert = 0.0


# ============================================================
# CAMERA CONTROLS
# ============================================================

left, right = st.columns(2)

with left:
    if st.button(
        "Start monitoring",
        type="primary",
    ):
        if (
            source_type == "IP camera"
            and not stream_url
        ):
            st.error(
                "Enter the IP camera stream URL first."
            )

        else:
            if st.session_state.capture is not None:
                st.session_state.capture.release()

            source = (
                int(camera_index)
                if source_type == "Laptop webcam"
                else stream_url
            )

            cap = cv2.VideoCapture(source)

            if cap.isOpened():
                st.session_state.capture = cap
                st.session_state.running = True

            else:
                st.session_state.capture = None
                st.session_state.running = False

                st.error(
                    "Could not open the camera. "
                    "Check camera permissions and source."
                )

with right:
    if st.button("Stop monitoring"):
        st.session_state.running = False

        if st.session_state.capture is not None:
            st.session_state.capture.release()

        st.session_state.capture = None


# ============================================================
# DASHBOARD OUTPUT
# ============================================================

preview = st.empty()
status = st.empty()
notification = st.empty()


# ============================================================
# LIVE PERSON + WEAPON DETECTION
# ============================================================

@st.fragment(run_every="0.3s")
def monitor():

    if (
        not st.session_state.running
        or st.session_state.capture is None
    ):
        status.info("Monitoring is stopped.")
        return

    cap = st.session_state.capture

    ok, frame = cap.read()

    if not ok:
        status.error(
            "Camera frame unavailable. "
            "Stop and restart monitoring."
        )
        return

    person_model, weapon_model = load_models()

    # Person detection
    persons = person_model.predict(
        frame,
        classes=[0],
        conf=person_threshold,
        imgsz=320,
        verbose=False,
    )[0]

    # Weapon detection
    weapons = None

    if weapon_model is not None:
        weapons = weapon_model.predict(
            frame,
            conf=weapon_threshold,
            imgsz=320,
            verbose=False,
        )[0]

    # --------------------------------------------------------
    # PERSON DETECTION RESULTS
    # --------------------------------------------------------

    person_count = 0
    person_confidences = []

    for box in persons.boxes:
        person_count += 1

        confidence = float(box.conf[0])
        person_confidences.append(confidence)

        draw_box(
            frame,
            box,
            "Person",
            (40, 200, 40),
        )

    # --------------------------------------------------------
    # WEAPON DETECTION RESULTS
    # --------------------------------------------------------

    weapon_count = 0
    detected_weapons = []
    weapon_confidences = []

    if weapons is not None:

        for box in weapons.boxes:
            weapon_count += 1

            weapon_name = weapons.names[
                int(box.cls[0])
            ]

            confidence = float(box.conf[0])

            detected_weapons.append(
                weapon_name
            )

            weapon_confidences.append(
                confidence
            )

            draw_box(
                frame,
                box,
                weapon_name,
                (20, 20, 240),
            )

    # --------------------------------------------------------
    # MONITORING STATUS
    # --------------------------------------------------------

    active = within_hours(
        start_hour,
        end_hour,
        datetime.now().time(),
    )

    preview.image(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        ),
        channels="RGB",
        use_container_width=True,
    )

    status.write(
        f"Security schedule: "
        f"{'ACTIVE' if active else 'INACTIVE'}"
        f" | People: {person_count}"
        f" | Weapon candidates: {weapon_count}"
    )

    now = time.monotonic()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    source_name = (
        "Webcam"
        if source_type == "Laptop webcam"
        else "IP camera"
    )

    # --------------------------------------------------------
    # WEAPON ALERT â€” HIGHEST PRIORITY
    # --------------------------------------------------------

    if active and weapon_count > 0:

        weapon_names = sorted(
            set(detected_weapons)
        )

        weapon_text = ", ".join(
            weapon_names
        )

        highest_weapon_confidence = max(
            weapon_confidences,
            default=0.0,
        )

        if (
            now - st.session_state.last_weapon_alert
            >= weapon_cooldown
        ):

            event = (
                "Possible weapon detected: "
                f"{weapon_text}"
            )

            log_event(
                event,
                highest_weapon_confidence,
                source_name,
            )

            st.session_state.last_weapon_alert = now

            notification.warning(
                f"{event} at {timestamp}. "
                "Human verification required."
            )

            if email_enabled:

                subject = (
                    "YOLO11 SECURITY ALERT: "
                    f"Possible {weapon_text} detected"
                )

                body = (
                    "YOLO11 HOME SECURITY ALERT\n\n"
                    f"Time: {timestamp}\n"
                    f"Camera: {source_name}\n"
                    f"People detected: {person_count}\n"
                    f"Possible weapon detected: "
                    f"{weapon_text}\n"
                    f"Highest confidence: "
                    f"{highest_weapon_confidence:.2f}\n\n"
                    "This is an AI-generated detection "
                    "and may be incorrect.\n"
                    "Please verify the camera feed.\n"
                    "This alert is not proof of a threat."
                )

                result = send_email(
                    subject,
                    body,
                )

                notification.info(result)

    # --------------------------------------------------------
    # PERSON ALERT â€” ONLY WHEN NO WEAPON IS DETECTED
    # --------------------------------------------------------

    elif active and person_count > 0:

        if (
            now - st.session_state.last_person_alert
            >= person_cooldown
        ):

            highest_person_confidence = max(
                person_confidences,
                default=0.0,
            )

            event = (
                "Person detected during monitoring hours"
            )

            log_event(
                event,
                highest_person_confidence,
                source_name,
            )

            st.session_state.last_person_alert = now

            notification.warning(
                f"{event} at {timestamp}"
            )

            if email_enabled:

                subject = (
                    "YOLO11 Home Security: "
                    "Person Detected"
                )

                body = (
                    "YOLO11 HOME SECURITY ALERT\n\n"
                    f"Time: {timestamp}\n"
                    f"Camera: {source_name}\n"
                    f"People detected: {person_count}\n"
                    "Possible weapons detected: "
                    "None in this frame\n\n"
                    "Please verify the camera feed. "
                    "This is not proof of an intrusion."
                )

                result = send_email(
                    subject,
                    body,
                )

                notification.info(result)


# Run monitoring
monitor()


# ============================================================
# DETECTION HISTORY
# ============================================================

st.subheader("Detection History")

if LOG.exists():

    import pandas as pd

    st.dataframe(
        pd.read_csv(LOG)
        .tail(50)
        .iloc[::-1],
        use_container_width=True,
    )

    st.download_button(
        "Download CSV log",
        LOG.read_bytes(),
        file_name="home_security_log.csv",
        mime="text/csv",
    )

else:
    st.caption("No events logged yet.")


# ============================================================
# MODEL STATUS
# ============================================================

if not WEAPON_WEIGHTS.exists():

    st.warning(
        "Custom weapon model not found at: "
        f"{WEAPON_WEIGHTS}. "
        "Person detection will still work."
    )


# ============================================================
# FAMILY REGISTRATION
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "Family Management"
)

st.sidebar.page_link(
    "pages/Family_Registration.py",
    label="Open Family Registration",
    icon=":material/home:",
)

