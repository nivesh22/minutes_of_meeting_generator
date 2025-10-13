"""
Email utility script to automate sending reports with an Excel attachment.

I wrote this code for production use in automation workflows where reports
stored as dataframes need to be exported and shared as Excel reports via email. 

It includes:
- Email validation before sending
- Attachment generation directly from a pandas DataFrame
- Reusable send_mail function with clear type hints

I like to use 'ruff' as a linter and formatter so that any code I write always 
stays neat, consistent, and readable.
"""

import re
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
from smtplib import SMTP

import pandas as pd

SERVER = ""  # Removed for privacy


def send_mail(
    sender: str,
    recipients_list: list[str],
    cc_list: list[str],
    subject: str,
    body: str,
    data_frame: pd.DataFrame,
    attachment_name: str,
) -> None:
    """
    Send mail to the users with the attachment file.

    Args:
        sender (str): Sender email id.
        recipients_list (list[str]): List of recipient email ids.
        cc_list (list[str]): List of CC email ids.
        subject (str): Mail subject.
        body (str): Mail body.
        data_frame (pd.DataFrame): Dataframe to be attached as Excel file.
        attachment_name (str): Attachment file name.
    """
    to = ",".join(recipients_list)
    cc = ",".join(cc_list)

    if not _check_valid_mail(sender):
        raise ValueError(f"Invalid sender email: {sender}")

    msg = MIMEMultipart()
    msg["To"] = to
    msg["From"] = sender
    msg["Subject"] = subject
    msg["Cc"] = cc
    msg.attach(MIMEText(body, "html"))

    if not data_frame.empty:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            data_frame.to_excel(writer, sheet_name="Data", index=False)
        output.seek(0)

        part = MIMEBase(
            "application",
            "vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        part.set_payload(output.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={attachment_name}",
        )
        msg.attach(part)

    with SMTP(SERVER) as smtp:
        smtp.send_message(msg)


def _check_valid_mail(email: str) -> bool:
    """Validate an email address using regex."""
    regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(regex, email) is not None
