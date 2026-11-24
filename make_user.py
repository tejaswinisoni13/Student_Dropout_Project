"""
Create an EduLens account entry for .streamlit/secrets.toml.

Usage:   python make_user.py
It asks for a username, display name, role(s) and password (typed twice, hidden),
then prints a TOML block containing only a salted PBKDF2-SHA256 hash. The password
itself is never stored or printed. Paste the block into .streamlit/secrets.toml.
"""
import getpass
import hashlib
import os
import re
import sys

ROLES = ["Faculty Member", "Course Coordinator"]   # keep in sync with ROLES in app.py
ITERATIONS = 600_000


def make_hash(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${digest.hex()}"


def main():
    username = input("Username (letters, digits, underscore): ").strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{3,32}", username):
        sys.exit("Username must be 3-32 characters: lowercase letters, digits or underscore.")
    name = input("Display name (e.g. Prof. Sharma): ").strip() or username
    print("Roles:", ", ".join(f"{i + 1} = {r}" for i, r in enumerate(ROLES)))
    picks = input("Role numbers for this account, comma separated [1]: ").strip() or "1"
    try:
        roles = [ROLES[int(p) - 1] for p in picks.split(",")]
    except (ValueError, IndexError):
        sys.exit("Invalid role selection.")
    pw = getpass.getpass("Password (min 10 characters): ")
    if len(pw) < 10:
        sys.exit("Password must be at least 10 characters.")
    if pw != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    role_list = ", ".join(f'"{r}"' for r in roles)
    print("\n# Paste this into .streamlit/secrets.toml (keep one [edulens.users.<name>] block per account)\n")
    print(f"[edulens.users.{username}]")
    print(f'name = "{name}"')
    print(f"roles = [{role_list}]")
    print(f'password_hash = "{make_hash(pw)}"')


if __name__ == "__main__":
    main()
