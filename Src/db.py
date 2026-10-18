import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Local .env use karo, lekin existing environment variables overwrite mat karo.
load_dotenv(ROOT / ".env", override=False)


def get_connection():
    database_url = os.getenv("DATABASE_URL")

    # Streamlit Cloud Secrets se URL read karo.
    if not database_url:
        try:
            import streamlit as st
            database_url = st.secrets.get("DATABASE_URL")
        except Exception:
            database_url = None

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL missing. Configure it in Streamlit Secrets."
        )

    return psycopg.connect(database_url)


def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(254) NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role VARCHAR(20) NOT NULL DEFAULT 'customer'
                        CHECK (role IN ('customer', 'admin')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)