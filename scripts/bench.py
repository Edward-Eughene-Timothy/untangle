"""Where does the time go? Talks straight to Ollama (the app doesn't need to be running).

Usage:  python scripts/bench.py                      # tests gemma4:e2b-it-qat
        python scripts/bench.py gemma3:1b-it-qat     # test any other installed model
"""
import sys

import httpx

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e2b-it-qat"
URL = "http://127.0.0.1:11434/api/chat"
MESSAGES = [{"role": "user", "content": "I've been feeling stuck about a decision at work."}]


def run(label: str, extra: dict, report: bool = True) -> None:
    body = {
        "model": MODEL, "stream": False, "messages": MESSAGES,
        "options": {"num_ctx": 2048, "num_predict": 120}, **extra,
    }
    r = httpx.post(URL, json=body, timeout=900)
    if r.status_code != 200:
        print(f"{label}: HTTP {r.status_code} {r.text[:150]}")
        return
    if not report:
        return
    d = r.json()
    m = d.get("message", {})
    sec = lambda k: d.get(k, 0) / 1e9
    p_tok, g_tok = d.get("prompt_eval_count", 0), d.get("eval_count", 0)
    print(
        f"{label}: total {sec('total_duration'):.1f}s | model load {sec('load_duration'):.1f}s | "
        f"reading prompt: {p_tok} tok in {sec('prompt_eval_duration'):.1f}s | "
        f"writing: {g_tok} tok in {sec('eval_duration'):.1f}s "
        f"({g_tok / max(sec('eval_duration'), 0.001):.1f} tok/s) | "
        f"hidden thinking: {len(m.get('thinking') or '')} chars"
    )


print(f"Model: {MODEL}  (first call loads the model; results below are the warm runs)")
run("warm-up", {}, report=False)
run("default    ", {})
run("think=false", {"think": False})
