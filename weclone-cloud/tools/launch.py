#!/usr/bin/env python3
"""Start weclone-cloud (persona API) and LangBot in this one console window.

* Both children share this console, so closing the window or pressing Ctrl+C
  terminates both of them.
* Output of both processes is streamed here with ``[weclone]`` / ``[langbot]``
  prefixes, so you see everything in one place.
* The persona service is started first; LangBot is launched only after
  ``/healthz`` answers, so it never fails on a cold start.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def out(tag: str, line: str) -> None:
    print(f"[{tag}] {line}", flush=True)


def pump(tag: str, stream, log_path: Path) -> None:
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            for raw in iter(stream.readline, ""):
                line = raw.rstrip("\r\n")
                handle.write(line + "\n")
                handle.flush()
                out(tag, line)
    except Exception as exc:  # stream closed during shutdown
        out(tag, f"<reader stopped: {exc}>")
    finally:
        try:
            stream.close()
        except Exception:
            pass


def wait_http(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.5)
    return False


def find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    local = ROOT / ".venv" / "Scripts" / "uv.exe"
    if local.is_file():
        return str(local)
    local_posix = ROOT / ".venv" / "bin" / "uv"
    if local_posix.is_file():
        return str(local_posix)
    return None


def writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return False, f"mkdir: {exc}"
    probe = path / ".write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
    except Exception as exc:
        return False, f"write: {exc}"
    try:
        probe.unlink()
    except Exception:
        pass
    return True, ""


def choose_langbot_dir(explicit: str) -> Path:
    if explicit:
        target = Path(explicit).expanduser()
        ok, err = writable(target)
        if not ok:
            raise SystemExit(f"cannot use {target}: {err}")
        return target

    candidates: list[Path] = []
    for var in ("LOCALAPPDATA", "APPDATA", "USERPROFILE", "TEMP"):
        base = os.environ.get(var)
        if base:
            candidates.append(Path(base) / "weclone-langbot")
    # ChromaDB chokes on paths under Documents (OneDrive / Controlled Folder
    # Access); a plain short path at the drive root is the last resort.
    for drive in ("C:", "D:", "E:"):
        if Path(f"{drive}/").exists():
            candidates.append(Path(f"{drive}/langbot"))
            break
    candidates.append(ROOT / ".langbot-data")

    for candidate in candidates:
        ok, err = writable(candidate)
        if ok:
            return candidate
        out("langbot", f"skip {candidate}: {err}")
    raise SystemExit("no writable directory for LangBot data; pass --langbot-dir <path>")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--persona", default="persona_system.md")
    parser.add_argument("--no-langbot", action="store_true")
    parser.add_argument(
        "--langbot-dir",
        default="",
        help="LangBot data dir; defaults to %%LOCALAPPDATA%%\\weclone-langbot "
        "(Documents/OneDrive folders break ChromaDB with 'unable to open database file')",
    )
    parser.add_argument("--wait", type=float, default=40.0, help="seconds to wait for /healthz")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not (ROOT / "cloud.jsonc").is_file():
        print("cloud.jsonc missing. Run run-all.ps1 -Mode persona first.")
        return 1
    if not (ROOT / "app" / "weclone").is_dir():
        print("app/ missing. Run bootstrap first.")
        return 1

    persona = Path(args.persona)
    if not persona.is_absolute():
        persona = ROOT / persona
    if not persona.is_file():
        print(f"!! persona file not found: {persona}")
        print("   the service will fall back to serve.system_prompt in cloud.jsonc")

    # Environment for both children. A fresh cmd window has none of the variables
    # that run-all.ps1 had set, so provide sane defaults and ask for the key.
    env = os.environ.copy()
    env.setdefault("WECLONE_SERVE_KEY", "local-key")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # The global uv cache can be locked by another process or owned by an
    # elevated user ("Could not acquire lock ... os error 5"). Keep it local.
    env["UV_CACHE_DIR"] = str(ROOT / ".uv-cache")
    if not env.get("WECLONE_API_KEY"):
        print("DeepSeek API key is not set in this window.")
        try:
            key = getpass.getpass("   paste it here (input hidden): ").strip()
        except Exception:
            key = ""
        if key:
            env["WECLONE_API_KEY"] = key
            os.environ["WECLONE_API_KEY"] = key
        else:
            print("   !! no key: model requests will fail with 401")
    os.environ["WECLONE_SERVE_KEY"] = env.get("WECLONE_SERVE_KEY", "local-key")

    print("=" * 66)
    print(" WeClone persona + LangBot")
    print(f"   persona API : http://{args.host}:{args.port}/v1")
    print(f"   web chat    : http://{args.host}:{args.port}/")
    print("   LangBot UI  : http://localhost:5300")
    print("   close this window (or Ctrl+C) to stop everything")
    if args.host not in ("127.0.0.1", "localhost"):
        print("")
        print("   NOTE: bound to all interfaces (needed by Docker / other machines)")
        print(f"   serve key : {env.get('WECLONE_SERVE_KEY', '')}   <- paste this into LangBot")
        print("   keep it private; stop the window when you are done")
    print("=" * 66)
    print()

    processes: list[tuple[str, subprocess.Popen]] = []
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    def spawn(tag: str, cmd: list[str], cwd: Path) -> subprocess.Popen:
        out(tag, "starting: " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        log_path = log_dir / f"{tag}.log"
        log_path.write_text("", encoding="utf-8")
        threading.Thread(target=pump, args=(tag, proc.stdout, log_path), daemon=True).start()
        processes.append((tag, proc))
        return proc

    weclone_cmd = [
        sys.executable,
        "-m",
        "weclone_cloud.cli",
        "serve",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if persona.is_file():
        weclone_cmd += ["--persona", str(persona)]

    try:
        spawn("weclone", weclone_cmd, ROOT)

        health = f"http://{args.host}:{args.port}/healthz"
        if not wait_http(health, args.wait):
            out("weclone", f"!! {health} did not answer in {args.wait:.0f}s; check the log above")
        else:
            out("weclone", "ready")

        if not args.no_langbot:
            uv = find_uv()
            if not uv:
                out(
                    "langbot", "!! uv not found, skipping LangBot (install uv or run start-langbot.cmd after)"
                )
            else:
                try:
                    data_dir = choose_langbot_dir(args.langbot_dir)
                except (SystemExit, Exception) as exc:
                    data_dir = None
                    out(
                        "langbot",
                        f"!! cannot prepare data dir ({exc}); skipping LangBot, persona API stays up",
                    )
                if data_dir is not None:
                    out("langbot", f"data dir: {data_dir}")
                    spawn("langbot", [uv, "tool", "run", "langbot@latest"], data_dir)

        while True:
            for entry in list(processes):
                tag, proc = entry
                code = proc.poll()
                if code is None:
                    continue
                out(tag, f"exited with code {code}")
                processes.remove(entry)
                if tag == "weclone":
                    out("all", "persona API stopped, shutting everything down")
                    return 0
                out("all", f"{tag} stopped; the persona API keeps running (close the window to stop it)")
            if not processes:
                return 0
            time.sleep(1.0)
    except KeyboardInterrupt:
        print()
        out("all", "Ctrl+C received, stopping...")
        return 0
    finally:
        for tag, proc in processes:
            if proc.poll() is None:
                out("all", f"terminating {tag}")
                try:
                    proc.terminate()
                except Exception:
                    pass
        deadline = time.time() + 8
        for tag, proc in processes:
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except Exception:
                out("all", f"killing {tag}")
                try:
                    proc.kill()
                except Exception:
                    pass
        out("all", "stopped")


if __name__ == "__main__":
    sys.exit(main())
