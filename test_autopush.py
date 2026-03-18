#!/usr/bin/env python3
"""Test auto-push without running a full agent step.
Usage:
  1. Make a small change (e.g. echo 'test' >> README.md)
  2. echo "test: auto-push" > .autopush
  3. python3 test_autopush.py
"""
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

WORKING_DIR = Path(__file__).parent
load_dotenv(WORKING_DIR / ".env")

AUTO_PUSH = os.environ.get("AUTO_PUSH", "").lower() in ("1", "true", "yes")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
AUTO_PUSH_REMOTE = os.environ.get("AUTO_PUSH_REMOTE", "origin")
AUTO_PUSH_BRANCH = os.environ.get("AUTO_PUSH_BRANCH", "main")

autopush_flag = WORKING_DIR / ".autopush"
if not autopush_flag.exists():
    print("Create .autopush first: echo 'test: auto-push' > .autopush")
    exit(1)

if not AUTO_PUSH:
    print("AUTO_PUSH is not enabled in .env")
    exit(1)

flag_msg = autopush_flag.read_text().strip()
autopush_flag.unlink(missing_ok=True)

print("Running git add...")
subprocess.run(
    ["git", "add", "-A", "--", ".", ":!.env", ":!.env.*", ":!context/", ":!logs/"],
    cwd=WORKING_DIR, capture_output=True, text=True, timeout=60,
)

status = subprocess.run(
    ["git", "diff", "--cached", "--stat"],
    cwd=WORKING_DIR, capture_output=True, text=True, timeout=10,
)
if not status.stdout.strip():
    print("Nothing to commit")
    exit(0)

msg = flag_msg or "test: auto-push"
print(f"Committing: {msg}")
r = subprocess.run(
    ["git", "commit", "-m", msg],
    cwd=WORKING_DIR, capture_output=True, text=True, timeout=15,
)
if r.returncode != 0:
    print(f"Commit failed: {r.stderr}")
    exit(1)

remote_url = subprocess.run(
    ["git", "remote", "get-url", AUTO_PUSH_REMOTE],
    cwd=WORKING_DIR, capture_output=True, text=True, timeout=5,
).stdout.strip()

if GITHUB_TOKEN and remote_url.startswith("https://github.com/"):
    auth_url = remote_url.replace(
        "https://github.com/", f"https://x-access-token:{GITHUB_TOKEN}@github.com/"
    )
    push_target = [auth_url, AUTO_PUSH_BRANCH]
else:
    push_target = [AUTO_PUSH_REMOTE, AUTO_PUSH_BRANCH]

print("Pushing...")
push_env = os.environ.copy()
if GITHUB_TOKEN:
    push_env["GIT_ASKPASS"] = "echo"
    push_env["GIT_TERMINAL_PROMPT"] = "0"

r = subprocess.run(
    ["git", "push"] + push_target,
    cwd=WORKING_DIR, capture_output=True, text=True, timeout=30, env=push_env,
)
if r.returncode != 0:
    print(f"Push failed: {r.stderr}")
    exit(1)

print(f"Done: pushed {msg}")
