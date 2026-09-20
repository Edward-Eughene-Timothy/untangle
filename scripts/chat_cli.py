"""Terminal client for Untangle (start the server first).

Usage:     python scripts/chat_cli.py
Requires:  pip install prompt_toolkit      (or: pip install -e ".[cli]")

- Paste any amount of text (even with line breaks): it appears as ONE message, and nothing is sent
  until you press Enter.
- Anything you type while the bot is replying is ignored (not echoed, not queued).
- Ctrl+C while the bot is replying cancels that reply; Ctrl+C or Ctrl+D at the prompt exits.
- /new starts a fresh conversation, /quit exits.
"""
import contextlib
import json
import sys

import httpx

try:
    from prompt_toolkit import prompt
except ImportError:
    sys.exit("Missing dependency. Run:  pip install prompt_toolkit")

try:  # POSIX only; on Windows typing during a reply just isn't suppressed
    import termios
except ImportError:
    termios = None

URL = "http://127.0.0.1:8000/api/v1/stream_counsel"
MAX_CHARS = 4000  # server limit


def read_message() -> str:
    return prompt("\nyou> ").strip()


@contextlib.contextmanager
def ignore_typing():
    """Turn off echo while the bot replies, and discard anything typed meanwhile."""
    if termios is None or not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    quiet = termios.tcgetattr(fd)
    quiet[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSADRAIN, quiet)
    try:
        yield
    finally:
        termios.tcflush(fd, termios.TCIFLUSH)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def stream_reply(message: str, sid: str | None) -> str | None:
    body = {"message": message, **({"session_id": sid} if sid else {})}
    event = None
    try:
        with ignore_typing(), httpx.stream("POST", URL, json=body, timeout=None) as r:
            if r.status_code != 200:
                r.read()
                print("HTTP", r.status_code, r.text)
                return sid
            for line in r.iter_lines():
                if line.startswith("event:"):
                    event = line[7:]
                elif line.startswith("data:"):
                    d = json.loads(line[6:])
                    if event == "meta":
                        sid = d["session_id"]
                        print("bot> ", end="", flush=True)
                    elif event == "safety":
                        print(f"\n[!] {d['message']}\n", end="")
                    elif event == "token":
                        print(d["text"], end="", flush=True)
                    elif event == "done":
                        info = f"first token {d['ttft_ms']} ms"
                        llm = d.get("llm") or {}
                        if llm.get("prompt_tokens") is not None:
                            info += f" | prompt {llm['prompt_tokens']} tok read in {llm['prompt_s']} s"
                        if llm.get("gen_tok_per_s"):
                            info += f" | wrote {llm['gen_tokens']} tok at {llm['gen_tok_per_s']} tok/s"
                        print(f"\n[{info}]")
                    elif event == "error":
                        print(f"\n[error] {d['message']}")
    except KeyboardInterrupt:
        print("\n[reply cancelled]")
    except httpx.HTTPError as exc:
        print(f"\n[connection problem: {exc}]")
    return sid


def main() -> None:
    sid = None
    print("Type or paste a message, then press Enter. /new = new conversation, /quit = exit.")
    while True:
        try:
            msg = read_message()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not msg:
            continue
        if msg == "/quit":
            return
        if msg == "/new":
            sid = None
            print("[new conversation]")
            continue
        if len(msg) > MAX_CHARS:
            print(f"[message was {len(msg)} characters; sending the first {MAX_CHARS}]")
            msg = msg[:MAX_CHARS]
        sid = stream_reply(msg, sid)


if __name__ == "__main__":
    main()
