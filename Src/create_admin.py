
import getpass

import bcrypt

from db import get_connection, init_db


def main():
    init_db()

    name = input("Admin name: ").strip()
    email = input("Admin email: ").strip().lower()
    password = getpass.getpass("Admin password (12+ chars): ")
    confirm = getpass.getpass("Confirm password: ")

    if not name or len(name) > 100:
        print("Invalid name.")
        return

    if len(password) < 12:
        print("Password must be at least 12 characters.")
        return

    if password != confirm:
        print("Passwords do not match.")
        return

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
                    VALUES (%s, %s, %s, 'admin')
                    """,
                    (name, email, password_hash)
                )

        print("Admin account created successfully.")

    except Exception:
        print("Could not create admin. Check the email and database.")


if __name__ == "__main__":
    main()
