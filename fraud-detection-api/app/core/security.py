"""
Security utilities:
  - API key dependency for FastAPI routes
  - IP address masking filter for the logging system
"""

import logging
import os
import re
import secrets

from fastapi import Header, HTTPException, status

# ── API Key ───────────────────────────────────────────────────────────────────

_API_KEY: str = os.getenv("API_KEY", "")

_IP_RE = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"          # IPv4
    r"|"
    r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"  # IPv6 (simplified)
)


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """
    FastAPI dependency — validates the X-API-Key header.

    Raises 401 if:
      - API_KEY env var is set and the header is missing or wrong
      - Uses constant-time comparison to prevent timing attacks

    If API_KEY is not configured (empty string), auth is disabled so
    the server is still usable in development without extra setup.
    """
    if not _API_KEY:
        return  # auth disabled in dev when API_KEY is unset

    # secrets.compare_digest prevents timing-based key enumeration
    if not x_api_key or not secrets.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "unauthorized",
                "message": "Missing or invalid X-API-Key header.",
                "detail": "Provide a valid API key in the X-API-Key request header.",
            },
        )


# ── IP masking log filter ─────────────────────────────────────────────────────

class _IPMaskFilter(logging.Filter):
    """
    Logging filter that replaces IP addresses in log records with [MASKED].
    Attached to the root logger so it covers every logger in the application.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _mask_ip(str(record.msg))
        record.args = _mask_args(record.args)
        return True


def _mask_ip(text: str) -> str:
    return _IP_RE.sub("[MASKED]", text)


def _mask_args(args: object) -> object:
    if isinstance(args, tuple):
        return tuple(_mask_ip(str(a)) if _IP_RE.search(str(a)) else a for a in args)
    if isinstance(args, dict):
        return {k: _mask_ip(str(v)) if _IP_RE.search(str(v)) else v for k, v in args.items()}
    return args


def install_ip_mask_filter() -> None:
    """Attach the IP masking filter to the root logger once at startup."""
    root = logging.getLogger()
    for f in root.filters:
        if isinstance(f, _IPMaskFilter):
            return  # already installed
    root.addFilter(_IPMaskFilter())
