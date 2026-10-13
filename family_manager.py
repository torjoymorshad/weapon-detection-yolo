
import streamlit as st
import sqlite3
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
import io

# -------------------------------
# FILE LOCATIONS
# -------------------------------

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "family_members.db"
PHOTO_DIR = ROOT / "family_photos"

PHOTO_DIR.mkdir(exist_ok=True)


# -------------------------------
# DATABASE
# -------------------------------

def connect():
    return sqlite3.connect(DB_PATH)


def setup_database():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS family (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                pin_hash TEXT NOT NULL UNIQUE,
                checked_in_until TEXT
            )
        """)

        columns = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(family)"
            ).fetchall()
        ]

        if "photo_path" not in columns:
            conn.execute(
                "ALTER TABLE family ADD COLUMN photo_path TEXT"
            )


# -------------------------------
# PIN MANAGEMENT
# -------------------------------

def hash_pin(pin):
    return hashlib.sha256(pin.encode()).hexdigest()


def register_member(name, photo):
    with connect() as conn:

        for _ in range(30):
            pin = f"{secrets.randbelow(1000000):06d}"

            try:
                cursor = conn.execute(
                    """
                    INSERT INTO family
                    (name, pin_hash, photo_path)
                    VALUES (?, ?, ?)
                    """,
                    (name.strip(), hash_pin(pin), None)
                )

                member_id = cursor.lastrowid
                break

            except sqlite3.IntegrityError:
                continue
        else:
            raise RuntimeError("Could not create a unique PIN.")

        if photo is not None:
            try:
                image = Image.open(
                    io.BytesIO(photo.getvalue())
                )

                image = image.convert("RGB")
                image.thumbnail((600, 600))

                photo_path = PHOTO_DIR / f"member_{member_id}.jpg"

                image.save(
                    photo_path,
                    format="JPEG",
                    quality=85
                )

                conn.execute(
                    """
                    UPDATE family
                    SET photo_path = ?
                    WHERE id = ?
                    """,
                    (str(photo_path), member_id)
                )

            except Exception as error:
                st.warning(
                    f"Member registered, but photo could "
                    f"not be saved: {error}"
                )

    return pin


def check_in(pin):
    if not (len(pin) == 6 and pin.isdigit()):
        return False

    until = datetime.now() + timedelta(hours=8)

    with connect() as conn:
        cursor = conn.execute(
            """
            UPDATE family
            SET checked_in_until = ?
            WHERE pin_hash = ?
            """,
            (until.isoformat(), hash_pin(pin))
        )

        return cursor.rowcount > 0


# -------------------------------
# FAMILY MEMBER DATA
# -------------------------------

def get_members():
    with connect() as conn:
        return conn.execute(
            """
            SELECT id, name, photo_path, checked_in_until
            FROM family
            ORDER BY id DESC
            """
        ).fetchall()


def authorized_member_present():
    now = datetime.now()

    for _, _, _, until in get_members():
        if until:
            try:
                if datetime.fromisoformat(until) > now:
                    return True
            except ValueError:
                pass

    return False


# -------------------------------
# FAMILY DASHBOARD
# -------------------------------

def family_dashboard():

    st.set_page_config(
        page_title="Family Security Management",
        page_icon="🏠",
        layout="wide"
    )

    setup_database()

    st.title("🏠 Family Security Management")

    st.caption(
        "Register family members with a photo and private PIN."
    )

    st.info(
        "Photos are used for family profiles only. "
        "This version does not perform automatic "
        "facial recognition."
    )

    tab1, tab2, tab3 = st.tabs([
        "Register Family",
        "Family Check-In",
        "Family Members"
    ])

    # ---------------------------
    # TAB 1: REGISTRATION
    # ---------------------------

    with tab1:

        st.header("Register Family Member")

        with st.form("registration_form"):

            name = st.text_input(
                "Family member name"
            )

            photo = st.file_uploader(
                "Upload family member photo",
                type=["jpg", "jpeg", "png"]
            )

            submitted = st.form_submit_button(
                "Register Member"
            )

            if submitted:

                if not name.strip():
                    st.error("Please enter a name.")

                else:
                    try:
                        pin = register_member(name, photo)

                        st.success(
                            f"{name} registered successfully!"
                        )

                        st.warning(
                            f"Private PIN: {pin}\n\n"
                            "Save this PIN securely. "
                            "It will not be displayed again."
                        )

                    except Exception as error:
                        st.error(
                            f"Registration failed: {error}"
                        )

    # ---------------------------
    # TAB 2: CHECK-IN
    # ---------------------------

    with tab2:

        st.header("Family Member Check-In")

        st.write(
            "Enter your private PIN to check in for 8 hours."
        )

        with st.form("checkin_form"):

            pin = st.text_input(
                "6-digit PIN",
                type="password",
                max_chars=6
            )

            submitted = st.form_submit_button(
                "Check In"
            )

            if submitted:

                if check_in(pin):
                    st.success(
                        "Check-in successful! "
                        "Authorization recorded for 8 hours."
                    )

                else:
                    st.error("Invalid PIN.")

    # ---------------------------
    # TAB 3: MEMBER PROFILES
    # ---------------------------

    with tab3:

        st.header("Registered Family Members")

        members = get_members()

        if not members:
            st.info(
                "No family members registered yet."
            )

        for member_id, name, photo_path, until in members:

            with st.container(border=True):

                col1, col2 = st.columns([1, 3])

                with col1:

                    if photo_path and Path(photo_path).exists():
                        st.image(
                            photo_path,
                            width=160
                        )

                    else:
                        st.write("No profile photo")

                with col2:

                    st.subheader(name)

                    active = False

                    if until:
                        try:
                            active = (
                                datetime.fromisoformat(until)
                                > datetime.now()
                            )
                        except ValueError:
                            pass

                    if active:
                        st.success("Checked In")
                    else:
                        st.info("Not Checked In")

                    st.caption(
                        f"Member ID: {member_id}"
                    )

        st.divider()

        if authorized_member_present():
            st.success(
                "At least one family member is checked in."
            )

        else:
            st.warning(
                "No family members are currently checked in."
            )


# -------------------------------
# START APPLICATION
# -------------------------------

if __name__ == "__main__":
    family_dashboard()
