import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Known prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"disregard.*instructions",
    r"you are now",
    r"act as",
    r"forget.*previous",
    r"system prompt",
    r"jailbreak",
    r"override.*decision",
    r"unconditional.*approval",
    r"approve.*regardless",
    r"bypass.*policy",
    r"ignore.*policy",
    r"new instruction",
    r"end of prompt",
    r"\<\|.*\|\>",  # special tokens
    r"###.*instruction",
]


def detect_prompt_injection(text: str) -> tuple[bool, Optional[str]]:
    """
    Detect prompt injection attempts in free-text fields.
    Returns (is_injection, matched_pattern)
    """
    if not text:
        return False, None

    text_lower = text.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning(f"Prompt injection detected: pattern='{pattern}' in text='{text[:100]}'")
            return True, pattern

    return False, None


def sanitize_text(text: str, max_length: int = 200) -> str:
    """
    Sanitize free-text input:
    - Truncate to max length
    - Remove special characters that could affect LLM behavior
    - Strip leading/trailing whitespace
    """
    if not text:
        return text

    # Truncate
    text = text[:max_length]

    # Remove potential injection characters
    text = re.sub(r'[<>{}[\]\\]', '', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text.strip()


def validate_application_security(application: dict) -> tuple[bool, Optional[str]]:
    """
    Run security checks on loan application data.
    Returns (is_safe, error_message)
    """
    # Check free-text fields for injection
    free_text_fields = ["purpose", "home_ownership", "verification_status"]

    for field in free_text_fields:
        value = str(application.get(field, ""))
        is_injection, pattern = detect_prompt_injection(value)
        if is_injection:
            return False, f"Security violation in field '{field}': suspicious pattern detected"

    # Validate numeric ranges
    checks = [
        ("loan_amnt", 0, 40000),
        ("int_rate", 0, 31),
        ("dti", 0, 100),
        ("fico_range_low", 300, 850),
        ("fico_range_high", 300, 850),
        ("revol_util", 0, 100),
    ]

    for field, min_val, max_val in checks:
        value = application.get(field)
        if value is not None:
            if not (min_val <= float(value) <= max_val):
                return False, f"Value out of range for '{field}': {value}"

    return True, None