"""Tests for app.services.exceptions — the provider error hierarchy."""
from __future__ import annotations

import pytest

from app.services.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderModelError,
    ProviderUnavailableError,
)


def test_provider_error_default_provider_is_unknown():
    err = ProviderError("boom")
    assert err.provider == "unknown"
    assert str(err) == "boom"


def test_provider_error_accepts_provider_kwarg():
    err = ProviderError("boom", provider="lmstudio")
    assert err.provider == "lmstudio"


def test_subclasses_inherit_provider_attribute():
    for cls in (ProviderAuthError, ProviderUnavailableError, ProviderModelError):
        err = cls("x", provider="ollama")
        assert err.provider == "ollama"
        assert str(err) == "x"


def test_subclasses_are_provider_error_instances():
    assert isinstance(ProviderAuthError("x"), ProviderError)
    assert isinstance(ProviderUnavailableError("x"), ProviderError)
    assert isinstance(ProviderModelError("x"), ProviderError)


def test_can_be_caught_as_exception():
    with pytest.raises(Exception):
        raise ProviderAuthError("x")
