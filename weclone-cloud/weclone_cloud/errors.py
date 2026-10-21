"""Typed errors so the CLI can print actionable messages instead of tracebacks."""

from __future__ import annotations


class CloudError(RuntimeError):
    """Base class for every error this package raises on purpose."""


class ConfigError(CloudError):
    """cloud.jsonc is missing, malformed, or lacks a required value."""


class DatasetError(CloudError):
    """The chat dataset cannot be read or cannot be converted into fine-tune rows."""


class ApiError(CloudError):
    """The chat-completions endpoint returned an error (or was unreachable)."""

    def __init__(self, status: int, message: str, body: str = "") -> None:
        self.status = status
        self.body = body
        detail = f"[{status}] {message}" if status else message
        if body:
            detail = f"{detail}: {body[:500]}"
        super().__init__(detail)


class FineTuneError(CloudError):
    """The fine-tuning job could not be created, polled, or completed."""

    def __init__(self, message: str, *, job: dict | None = None) -> None:
        self.job = job or {}
        super().__init__(message)
