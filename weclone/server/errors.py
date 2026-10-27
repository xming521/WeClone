"""Errors the web UI shows. The UI words each one in its own language by `code`; `detail` keeps the Chinese text
for other clients. Errors without a code are shown as their `detail`."""

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from weclone.utils import secure_storage as secure


class ApiError(HTTPException):
    def __init__(self, status: int, code: str, detail: str):
        super().__init__(status, detail)
        self.code = code


def error_response(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail, "code": code}, status_code=status)


def security_response(exc: secure.SecurityError, status: int) -> JSONResponse:
    """A storage error. Only a locked store has a code: the other ones are rare faults whose text says what broke."""
    if isinstance(exc, secure.StorageLocked):
        return error_response(status, "storage_locked", str(exc))
    return JSONResponse({"detail": str(exc)}, status_code=status)


def install_errors(app):
    @app.exception_handler(ApiError)
    async def api_error(request, exc: ApiError):
        return error_response(exc.status_code, exc.code, exc.detail)
