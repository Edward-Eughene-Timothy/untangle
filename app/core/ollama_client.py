"""Thin async client for the local Ollama HTTP API (the only LLM the app ever talks to)."""
import json
import logging
from collections.abc import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised for any failure talking to the local model."""


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        keep_alive: str = "30m",
        temperature: float = 0.6,
        num_ctx: int = 4096,
        think: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.num_ctx = num_ctx
        # Reasoning models (e.g. Gemma 4) otherwise write hundreds of hidden "thinking" tokens before
        # the first visible word, which on a CPU means tens of seconds of silence. Non-thinking
        # models accept and ignore this flag.
        self.think = think
        # Long read timeout: the first request after a cold start has to load the model.
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            transport=transport,
            timeout=httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=10.0),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    def _payload(self, messages, *, stream, fmt=None, temperature=None, num_predict=None) -> dict:
        options = {
            "temperature": self.temperature if temperature is None else temperature,
            "num_ctx": self.num_ctx,
        }
        if num_predict:
            options["num_predict"] = num_predict
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if fmt is not None:
            body["format"] = fmt
        return body

    @staticmethod
    async def _error_text(resp: httpx.Response) -> str:
        await resp.aread()
        try:
            msg = resp.json().get("error") or resp.text
        except ValueError:
            msg = resp.text
        hint = " Try `ollama pull <model>`." if resp.status_code == 404 else ""
        return f"Ollama returned {resp.status_code}: {msg}.{hint}"

    async def stream_chat(
        self, messages: list[dict], stats: dict | None = None
    ) -> AsyncIterator[str]:
        """Yield assistant tokens as Ollama produces them. If `stats` is given, it is filled with
        Ollama's timing fields (prompt_eval_count, prompt_eval_duration, eval_count, ...) once the
        stream ends."""
        try:
            async with self._http.stream(
                "POST", "/api/chat", json=self._payload(messages, stream=True)
            ) as resp:
                if resp.status_code >= 400:
                    raise OllamaError(await self._error_text(resp))
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        raise OllamaError(str(data["error"]))
                    token = (data.get("message") or {}).get("content")
                    if token:
                        yield token
                    if data.get("done"):
                        if stats is not None:
                            for k in ("prompt_eval_count", "prompt_eval_duration", "eval_count",
                                      "eval_duration", "load_duration", "total_duration"):
                                stats[k] = data.get(k)
                        break
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaError("The local model timed out.") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

    async def complete(
        self,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> str:
        """Non-streaming completion. Pass `json_schema` for schema-constrained JSON output."""
        body = self._payload(
            messages,
            stream=False,
            fmt=json_schema,
            temperature=temperature,
            num_predict=num_predict,
        )
        try:
            resp = await self._http.post("/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise OllamaError(await self._error_text(resp))
        try:
            return resp.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError("Unexpected response from Ollama.") from exc

    async def health(self) -> dict:
        try:
            resp = await self._http.get("/api/tags", timeout=3.0)
            resp.raise_for_status()
            names = [m.get("name", "") for m in resp.json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            return {"reachable": False, "model": self.model, "model_available": False}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return {"reachable": True, "model": self.model, "model_available": wanted in names}

    async def preload(self) -> None:
        """Ask Ollama to load the model into memory so the first real request is fast."""
        try:
            await self._http.post(
                "/api/generate", json={"model": self.model, "keep_alive": self.keep_alive}
            )
        except httpx.HTTPError as exc:
            logger.info("Model preload skipped: %s", exc)
