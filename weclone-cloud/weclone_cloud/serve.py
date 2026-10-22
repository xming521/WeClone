"""Serve the fine-tuned persona as an OpenAI-compatible endpoint.

WeClone's own ``server --inference`` loads weights locally.  A cloud fine-tune
returns a *model id* instead, so serving is a thin, authenticated proxy:

    client -> /v1/chat/completions -> upstream (fine-tuned model + persona system)

That keeps existing clients working unchanged (AstrBot / LangBot / any OpenAI
SDK), because only the base URL and key have to be swapped.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config import CloudConfig
from .errors import ConfigError
from .finetune import load_state

CHAT_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WeClone Cloud Persona</title>
<style>
 body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 main{max-width:760px;margin:0 auto;padding:24px}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 .hint{color:#8b93a1;font-size:12px;margin-bottom:16px}
 #log{background:#171a21;border:1px solid #262b36;border-radius:10px;padding:14px;height:52vh;overflow:auto}
 .msg{margin:0 0 10px;white-space:pre-wrap;line-height:1.55}
 .me{color:#7cc4ff}.bot{color:#a6e3a1}
 form{display:flex;gap:8px;margin-top:12px}
 textarea{flex:1;resize:none;height:56px;background:#171a21;color:#e6e6e6;border:1px solid #262b36;border-radius:8px;padding:10px}
 button{background:#2f6feb;color:#fff;border:0;border-radius:8px;padding:0 18px;font-size:14px;cursor:pointer}
 button:disabled{opacity:.5;cursor:default}
</style></head><body><main>
<h1>数字分身 · 云端微调模型</h1>
<div class="hint" id="meta"></div>
<div id="log"></div>
<form id="f"><textarea id="q" placeholder="说点什么… (Enter 发送, Shift+Enter 换行)"></textarea><button id="send">发送</button></form>
<script>
const MODEL=__MODEL__;
const SYSTEM=__SYSTEM__;
const KEY=__KEY__;
const log=document.getElementById('log'),q=document.getElementById('q'),send=document.getElementById('send');
const HEADERS={'Content-Type':'application/json'};
if(KEY){HEADERS['Authorization']='Bearer '+KEY;}
document.getElementById('meta').textContent='model: '+MODEL+' · system: '+SYSTEM;
const history=[];
function add(cls,text){const p=document.createElement('p');p.className='msg '+cls;p.textContent=text;log.appendChild(p);log.scrollTop=log.scrollHeight;return p}
async function ask(){
  const text=q.value.trim(); if(!text||send.disabled) return;
  q.value=''; add('me','我: '+text); history.push({role:'user',content:text});
  send.disabled=true; const wait=add('bot','…');
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',headers:HEADERS,
      body:JSON.stringify({messages:history,stream:false})});
    const data=await r.json();
    if(!r.ok) throw new Error(data.detail||JSON.stringify(data));
    const reply=data.choices[0].message.content;
    history.push({role:'assistant',content:reply});
    wait.textContent='TA: '+reply;
  }catch(e){wait.textContent='[错误] '+e.message}
  finally{send.disabled=false; q.focus()}
}
send.onclick=(e)=>{e.preventDefault();ask()};
q.addEventListener('keydown',(e)=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}});
</script></body></html>"""


def resolve_model_id(config: CloudConfig) -> str:
    """微调模型 > 配置指定 > 回退到 api.model（DeepSeek 没微调时直接用人设提示词跑）。"""
    model = config.serve.model or load_state(config.finetune.state_file).model or config.api.model
    if not model:
        raise ConfigError("无法确定模型：serve.model、finetune_state.json、api.model 都是空的。")
    return model


def load_system_prompt(config: CloudConfig) -> str:
    """优先读 persona 文件（weclone-cloud persona 生成），否则用配置里的固定文本。"""
    target = config.serve.system_prompt_file
    if target:
        path = Path(target)
        if not path.is_file():
            raise ConfigError(f"serve.system_prompt_file 指向的文件不存在：{path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ConfigError(f"{path} 是空文件")
        return text
    return config.serve.system_prompt


def build_messages(
    messages: list[dict[str, Any]],
    *,
    system_prompt: str,
    mode: str = "replace",
) -> list[dict[str, Any]]:
    """Apply the persona system prompt to an incoming request.

    Chat frameworks such as LangBot send their own system message ("you are a
    helpful assistant"), which would silently override the persona.  ``replace``
    drops theirs; ``prepend`` keeps it after ours; ``if_missing`` only injects
    when the client sent none.
    """
    cleaned = [message for message in messages if isinstance(message, dict)]
    has_system = any(message.get("role") == "system" for message in cleaned)
    if mode == "if_missing" and has_system:
        return cleaned
    if mode == "replace":
        cleaned = [message for message in cleaned if message.get("role") != "system"]
    return [{"role": "system", "content": system_prompt}, *cleaned]


def create_app(
    config: CloudConfig,
    *,
    model_id: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    resolved_model = model_id or resolve_model_id(config)
    expected_key = config.serve.resolved_key
    system_prompt = load_system_prompt(config)
    # Fail at startup instead of returning 500 on every request: the upstream
    # key is only needed when a request arrives, so without this check the
    # service starts happily and breaks on the first message.
    try:
        config.api.resolved_key
    except ConfigError as exc:
        raise ConfigError(f"上游 API Key 不可用，服务无法工作：{exc}") from exc
    upstream = f"{config.api.base_url}/chat/completions"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient(
            timeout=httpx.Timeout(config.api.timeout, connect=15.0),
            transport=transport,
        )
        try:
            yield
        finally:
            await app.state.http.aclose()

    app = FastAPI(title="weclone-cloud persona", version="0.1.0", lifespan=lifespan)
    app.state.model_id = resolved_model

    def authorize(request: Request) -> None:
        if expected_key is None:
            return
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(token, expected_key):
            raise HTTPException(status_code=401, detail="invalid api key")

    def upstream_headers() -> dict[str, str]:
        try:
            key = config.api.resolved_key
        except ConfigError as exc:
            raise HTTPException(status_code=500, detail=f"上游 API Key 不可用：{exc}") from exc
        return {"Authorization": f"Bearer {key}"}

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "model": resolved_model}

    @app.get("/v1/models")
    def models(_: None = Depends(authorize)) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": resolved_model, "object": "model", "owned_by": "weclone-cloud"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict[str, Any], request: Request, _: None = Depends(authorize)):
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages 不能为空")

        payload = dict(body)
        payload["model"] = resolved_model
        payload["messages"] = build_messages(
            messages, system_prompt=system_prompt, mode=config.serve.system_mode
        )
        payload.setdefault("temperature", config.serve.temperature)
        payload.setdefault("top_p", config.serve.top_p)
        payload.setdefault("max_tokens", config.serve.max_tokens)
        # The proxy previously forwarded the client body untouched, so a
        # reasoning model kept thinking and callers (LangBot) rendered the
        # chain of thought into the chat. Disable it unless the client asked
        # for something explicitly.
        thinking = config.api.resolved_thinking
        if thinking in {"enabled", "disabled"}:
            payload.setdefault("thinking", {"type": thinking})

        http: httpx.AsyncClient = request.app.state.http
        # Human-like pause: bursty replies are the quickest way to get a
        # personal WeChat account rate-limited or banned.
        lo = max(0.0, config.serve.reply_delay_min)
        hi = max(lo, config.serve.reply_delay_max)
        if hi > 0:
            await asyncio.sleep(random.uniform(lo, hi))
        if payload.get("stream"):

            async def stream_body():
                try:
                    async with http.stream(
                        "POST", upstream, json=payload, headers=upstream_headers()
                    ) as response:
                        if response.status_code >= 400:
                            detail = (await response.aread()).decode("utf-8", "replace")[:500]
                            yield f"data: {json.dumps({'error': {'message': detail}}, ensure_ascii=False)}\n\n".encode()
                            return
                        buffer = b""
                        async for chunk in response.aiter_bytes():
                            buffer += chunk
                            while b"\n" in buffer:
                                line, buffer = buffer.split(b"\n", 1)
                                yield sanitize_sse_line(line) + b"\n"
                        if buffer:
                            yield sanitize_sse_line(buffer)
                except httpx.HTTPError as exc:
                    yield f"data: {json.dumps({'error': {'message': str(exc)}}, ensure_ascii=False)}\n\n".encode()

            return StreamingResponse(stream_body(), media_type="text/event-stream")

        try:
            response = await http.post(upstream, json=payload, headers=upstream_headers())
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"上游不可达：{exc}") from exc
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text[:1000])
        try:
            data = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="上游返回非 JSON") from exc
        strip_reasoning(data)
        return JSONResponse(data)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        page = CHAT_PAGE.replace("__MODEL__", _js(resolved_model))
        page = page.replace("__SYSTEM__", _js(system_prompt))
        page = page.replace("__KEY__", _js(expected_key or ""))
        return page

    return app


def strip_reasoning(data: Any) -> None:
    """Remove chain-of-thought fields a reasoning model may return.

    LangBot renders ``reasoning_content`` as part of the message, so a thinking
    model leaks an English ``<think>`` block into the chat. Dropping the field
    here keeps every downstream client clean.
    """
    if not isinstance(data, dict):
        return
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        for key in ("message", "delta"):
            part = choice.get(key)
            if isinstance(part, dict):
                part.pop("reasoning_content", None)
                part.pop("reasoning", None)


def sanitize_sse_line(line: bytes) -> bytes:
    """Strip reasoning from one ``data: {...}`` line of a stream."""
    text = line.decode("utf-8", "replace")
    stripped = text.lstrip()
    if not stripped.startswith("data:"):
        return line
    raw = stripped[5:].strip()
    if not raw or raw == "[DONE]":
        return line
    try:
        obj = json.loads(raw)
    except ValueError:
        return line
    strip_reasoning(obj)
    return ("data: " + json.dumps(obj, ensure_ascii=False)).encode("utf-8")


def _js(value: str) -> str:
    """JSON-encode for inline <script>, neutralising ``</script>``."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


# Prefer the standalone page under web/index.html when it exists; it is easier
# to restyle than a string literal and is picked up automatically on restart.
_PAGE_FILE = Path(__file__).with_name("web") / "index.html"
if _PAGE_FILE.is_file():
    CHAT_PAGE = _PAGE_FILE.read_text(encoding="utf-8")
