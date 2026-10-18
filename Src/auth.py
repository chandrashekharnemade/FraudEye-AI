
import re

import bcrypt
import streamlit as st

from db import get_connection, init_db


def normalize_email(email):
    return email.strip().lower()


def valid_email(email):
    return bool(
        re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            email.strip()
        )
    )


def create_customer(name, email, password):
    name = name.strip()
    email = normalize_email(email)

    if not name or len(name) > 100:
        return False, "Enter a valid name."

    if not valid_email(email):
        return False, "Enter a valid email address."

    if len(password) < 12:
        return False, "Password must be at least 12 characters."

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users
                        (name, email, password_hash, role)
                    VALUES (%s, %s, %s, 'customer')
                    """,
                    (name, email, password_hash)
                )

        return True, "Account created. Please log in."

    except Exception as exc:
        error_text = str(exc).lower()

        if "unique" in error_text or "duplicate" in error_text:
            return False, "This email is already registered."

        return False, "Account creation failed. Check database settings."


def verify_login(email, password):
    email = normalize_email(email)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, email, password_hash, role
                FROM users
                WHERE email = %s
                """,
                (email,)
            )
            user = cur.fetchone()

    if not user:
        return None

    user_id, name, saved_email, password_hash, role = user

    try:
        valid = bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8")
        )
    except (ValueError, TypeError, AttributeError):
        valid = False

    if not valid:
        return None

    return {
        "id": user_id,
        "name": name,
        "email": saved_email,
        "role": role,
    }


def logout():
    st.session_state.pop("authenticated", None)
    st.session_state.pop("user", None)


def render_auth_gate():
    init_db()

    if st.session_state.get("authenticated"):
        user = st.session_state.get("user", {})
        if user.get("role") in ("admin", "customer"):
            return True

        logout()

    st.title("🛡️ FraudEye AI")
    st.caption("Sign in to access your fraud detection dashboard.")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        with st.form("fraudeye_login_form"):
            email = st.text_input("Email address")
            password = st.text_input(
                "Password",
                type="password"
            )

            submitted = st.form_submit_button(
                "Login",
                use_container_width=True
            )

        if submitted:
            if not email.strip() or not password:
                st.error("Enter your email and password.")
            else:
                try:
                    user = verify_login(email, password)

                    if user:
                        if user["role"] not in ("admin", "customer"):
                            st.error("This account has an invalid role.")
                        else:
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = user
                            st.rerun()
                    else:
                        st.error("Invalid email or password.")

                except Exception:
                    st.error(
                        "Unable to log in. Check the database connection."
                    )

    with signup_tab:
        with st.form("fraudeye_signup_form"):
            name = st.text_input("Full name")
            new_email = st.text_input("Email address")
            new_password = st.text_input(
                "Password (minimum 12 characters)",
                type="password"
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password"
            )

            submitted = st.form_submit_button(
                "Create Customer Account",
                use_container_width=True
            )

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                try:
                    success, message = create_customer(
                        name,
                        new_email,
                        new_password
                    )

                    if success:
                        st.success(message)
                    else:
                        st.error(message)

                except Exception:
                    st.error(
                        "Unable to create account. Check database settings."
                    )

    return False


def render_role_info():
    user = st.session_state.get("user", {})

    st.sidebar.markdown("---")
    st.sidebar.write(f"👤 **{user.get('name', 'User')}**")
    st.sidebar.write(f"Role: **{user.get('role', 'customer').title()}**")

    if st.sidebar.button("Logout", use_container_width=True):
        logout()
        st.rerun()
