"""
Notification channels: Telegram and email (SMTP).
All credentials come from environment variables (populated from GitHub
Secrets in the Actions workflow) -- never hardcode tokens here.
"""
from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText

import requests


def send_telegram(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[notifiers] Telegram credentials missing, skipping Telegram send.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }, timeout=15)
    if not resp.ok:
        print(f"[notifiers] Telegram send failed: {resp.status_code} {resp.text}")


def send_email(subject: str, body: str) -> None:
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    to_addr = os.environ.get("ALERT_EMAIL_TO", user)

    if not all([host, user, password, to_addr]):
        print("[notifiers] Email credentials missing, skipping email send.")
        return

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())
    except Exception as exc:
        print(f"[notifiers] Email send failed: {exc}")
