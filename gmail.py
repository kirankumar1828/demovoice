import smtplib
import re
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_email(receiver, subject, body):
    sender_id = os.getenv("GMAIL_SENDER_ID")
    app_password = os.getenv("GMAIL_APP_PASSWORD")

    if not sender_id or not app_password:
        logger.error("Sender ID or App Password not found in environment variables.")
        return False

    msg = MIMEMultipart()
    msg["From"] = sender_id
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender_id, app_password)
            s.sendmail(sender_id, receiver, msg.as_string())
            logger.info("✅ Email sent successfully!")
            return True
    except smtplib.SMTPException as e:
        logger.error(f"❌ Email sending failed: {e}")
        return False

def sanitize_email(text):
    if not text:
        return None

    text = text.lower().replace(" at the rate ", "@").replace(" at ", "@").replace(" dot ", ".")
    text = text.replace(" underscore ", "_").replace(" dash ", "-")
    text = re.sub(r"\s+", "", text)  # Remove all spaces

    return text if check_email(text) else None

def check_email(email):

    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

# Example usage
if __name__ == "__main__":
    receiver = sanitize_email("example at gmail dot com")
    if receiver:
        send_email(receiver, "Test Subject", "This is a test email body.")
    else:
        logger.error("Invalid email address.")