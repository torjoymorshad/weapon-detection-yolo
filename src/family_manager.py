
if __name__ == "__main__":
    family_dashboard()

import streamlit as st
import sqlite3
import secrets
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "family_members.db"


def connect():
    return sqlite3.connect(DB_PATH)


def hash_pin(pin):
    return hashlib.sha256(pin.encode()).hexdigest()


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


def register_member(name):
    pin = str(secrets.randbelow(900000) + 100000)

    with connect() as conn:
        conn.execute(
            "INSERT INTO family (name, pin_hash) VALUES (?, ?)",
            (name.strip(), hash_pin(pin))
        )

    return pin


def check_in(pin):
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


def get_members():
    with connect() as conn:
        return conn.execute(
            "SELECT id, name, checked_in_until FROM family"
        ).fetchall()


def authorized_member_present():
    now = datetime.now()

    for _, _, checked_in_until in get_members():
        if checked_in_until:
            if datetime.fromisoformat(checked_in_until) > now:
                return True

    return False


def family_dashboard():
    setup_database()

    st.header("Family Member Registration")

    with st.form("register_family"):
        name = st.text_input("Family member name")
        submitted = st.form_submit_button("Register Member")

        if submitted:
            if not name.strip():
                st.error("Please enter a name.")
            else:
                pin = register_member(name)
                st.success(f"Registered {name} successfully!")
                st.warning(
                    f"Private PIN: {pin}. Save it now. "
                    "It will not be shown again."
                )

    st.divider()

    st.subheader("Family Check-In")

    with st.form("family_checkin"):
        pin = st.text_input(
            "Enter your 6-digit PIN",
            type="password",
            max_chars=6
        )
        checkin = st.form_submit_button("Check In")

        if checkin:
            if check_in(pin):
                st.success("Check-in successful for 8 hours!")
            else:
                st.error("Invalid PIN.")

    st.divider()

    st.subheader("Registered Members")

    members = get_members()

    if not members:
        st.info("No family members registered yet.")

    for _, name, until in members:
        active = (
            until is not None
            and datetime.fromisoformat(until) > datetime.now()
        )

        status = "Checked In" if active else "Not Checked In"
        st.write(f"**{name}** — {status}")
