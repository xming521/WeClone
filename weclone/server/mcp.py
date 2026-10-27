"""A minimal MCP server for the Streamable HTTP transport: tools only, stateless, one JSON body per request.

Each POST carries one JSON-RPC message. A request gets its response as the body. Notifications and responses get 202
without a body. The server keeps no session and opens no SSE stream, so clients get 405 for GET and DELETE.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

# Newest first. A client asking for any other version is offered the newest and decides whether to go on.
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26")
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict
    # Hints such as readOnlyHint, which clients may use to skip asking before a call.
    annotations: dict = field(default_factory=dict)


def error(message_id, code: int, text: str) -> dict:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": text}}


def result(message_id, value: dict) -> dict:
    return {"jsonrpc": "2.0", "id": message_id, "result": value}


def handle(
    message,
    *,
    server: dict,
    instructions: str,
    tools: list[Tool],
    call: Callable[[str, dict], tuple[str, bool]],
) -> dict | None:
    """The response to one JSON-RPC message, or None for a notification or a response. `call` runs a tool and
    returns its text and whether it failed."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return error(None, INVALID_REQUEST, "Expected a single JSON-RPC 2.0 message")
    # Responses answer server requests, which this server never sends.
    if "method" not in message or "id" not in message:
        return None
    message_id, method, params = message["id"], message["method"], message.get("params") or {}
    if not isinstance(params, dict):
        return error(message_id, INVALID_PARAMS, "params must be an object")
    if method == "initialize":
        requested = params.get("protocolVersion")
        return result(
            message_id,
            {
                "protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": server,
                "instructions": instructions,
            },
        )
    if method == "ping":
        return result(message_id, {})
    if method == "tools/list":
        listed = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                **({"annotations": tool.annotations} if tool.annotations else {}),
            }
            for tool in tools
        ]
        return result(message_id, {"tools": listed})
    if method == "tools/call":
        name, arguments = params.get("name"), params.get("arguments") or {}
        if name not in {tool.name for tool in tools}:
            return error(message_id, INVALID_PARAMS, f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            return error(message_id, INVALID_PARAMS, "arguments must be an object")
        text, failed = call(name, arguments)
        return result(message_id, {"content": [{"type": "text", "text": text}], "isError": failed})
    return error(message_id, METHOD_NOT_FOUND, f"Method not found: {method}")
