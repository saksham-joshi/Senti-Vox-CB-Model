import re

# --- Email
_LOCAL = r"A-Za-z0-9!#$%&'*+/=?^_`{|}~\-"
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"

_EMAIL = (
    r"(?:mailto:)?"
    rf"(?<![{_LOCAL}.])"
    rf"[{_LOCAL}]+(?:\.[{_LOCAL}]+)*"
    rf"@(?:{_LABEL}\.)+[A-Za-z]{{2,}}"
)

# --- URL 
_CH = r"[^\s()<>\"']"
_URL = (
    r"(?:(?:https?|ftps?)://|www\.)"
    rf"(?:{_CH}|\({_CH}*\))*"
    rf"(?:\({_CH}*\)|[^\s()<>\"'.,;:!?\]}}])"
)

_PATTERN = re.compile(rf"{_EMAIL}|{_URL}", re.IGNORECASE)
_SPACES = re.compile(r"[ \t]{2,}")


def remove_urls_and_emails(
    text: str,
    replacement: str = "",
    collapse_spaces: bool = False,
) -> str:
    """Remove (or replace) all URLs and email addresses in `text`."""
    if not text:
        return text

    if "@" in text or "://" in text or "www." in text.lower():
        text = _PATTERN.sub(replacement, text)

    if collapse_spaces:
        text = _SPACES.sub(" ", text).strip()
    return text