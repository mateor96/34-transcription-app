"""Thin wrapper over the OS keychain for storing the LLM provider API key.

On macOS this resolves to the login Keychain via the `keyring` library. Tests
substitute an in-memory backend (see tests/conftest.py).
"""
from __future__ import annotations

import keyring
import keyring.errors

SERVICE = "nota.ai"
USERNAME = "api_key"


def get_api_key() -> str:
    value = keyring.get_password(SERVICE, USERNAME)
    return value or ""


def set_api_key(value: str) -> None:
    if value:
        keyring.set_password(SERVICE, USERNAME, value)
    else:
        try:
            keyring.delete_password(SERVICE, USERNAME)
        except keyring.errors.PasswordDeleteError:
            pass
