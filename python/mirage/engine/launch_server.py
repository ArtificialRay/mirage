"""Launch the Mirage LLM Engine as an OpenAI-compatible HTTP server.

Usage::

    python -m mirage.engine.launch_server \\
        --model Qwen/Qwen3-8B \\
        --max-num-batched-requests 4 \\
        --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .model_runner import ModelRunner, RunnerConfig
from .llm_engine import LLMEngine


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: RunnerConfig = app.state.runner_config
    runner = ModelRunner(config)
    engine = LLMEngine(runner)
    app.state.engine = engine
    yield
    engine.close()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="MPK LLM Engine", lifespan=lifespan)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _parse_json(request: Request) -> dict:
    """Parse JSON body, returning 400 on empty or malformed input."""
    try:
        return await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or empty JSON body")


def _extract_prompt(messages: list[dict]) -> str:
    """Pull the last user message from an OpenAI chat messages list."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg["content"]
    return ""


async def _stream_bridge(
    engine: LLMEngine,
    prompt: str,
    mode: Literal["chat", "completions"] = "chat",
) -> AsyncGenerator[str, None]:
    """Bridge a synchronous streaming generator to async SSE chunks.

    Emits OpenAI-compatible chunks (`text` for completions, `delta.content` for
    chat) and a final `usage.completion_tokens` chunk before `data: [DONE]` so
    benchmark clients can read the server-reported output token count.
    """
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            for text, is_final in engine.submit(prompt, stream=True):
                queue.put_nowait((text, is_final, None))
        except Exception as exc:
            queue.put_nowait(("", True, str(exc)))

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, _run)

    n_tokens = 0
    while True:
        try:
            text, is_final, error = queue.get_nowait()
        except asyncio.QueueEmpty:
            if future.done():
                break
            await asyncio.sleep(0.01)
            continue

        if error:
            yield f"data: {{\"error\": \"{error}\"}}\n\n"
            break

        # Skip the synthetic ("", True) terminator yielded by LLMEngine when
        # the request completes with no new tokens — it carries no content.
        if not (text == "" and is_final):
            if mode == "completions":
                chunk = json.dumps({
                    "choices": [{"text": text, "index": 0}],
                })
            else:
                chunk = json.dumps({
                    "choices": [{"delta": {"content": text}, "index": 0}],
                })
            yield f"data: {chunk}\n\n"
            n_tokens += 1
        if is_final:
            break

    usage_chunk = json.dumps({
        "choices": [],
        "usage": {"completion_tokens": n_tokens},
    })
    yield f"data: {usage_chunk}\n\n"
    yield "data: [DONE]\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await _parse_json(request)
    prompt = _extract_prompt(body.get("messages", []))
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            _stream_bridge(request.app.state.engine, prompt, mode="chat"),
            media_type="text/event-stream",
        )
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: request.app.state.engine.submit(prompt),
        )
        return {
            "id": "chatcmpl-0",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }],
        }


@app.post("/v1/completions")
async def completions(request: Request):
    body = await _parse_json(request)
    prompt = body.get("prompt", "")
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            _stream_bridge(request.app.state.engine, prompt, mode="completions"),
            media_type="text/event-stream",
        )
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: request.app.state.engine.submit(prompt),
        )
        return {
            "id": "cmpl-0",
            "object": "text_completion",
            "choices": [{
                "index": 0,
                "text": result["text"],
                "finish_reason": "stop",
            }],
        }


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Mirage LLM Engine Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", default=8000, type=int, help="Port to listen on")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HuggingFace model name")
    parser.add_argument("--model-path", default=None, help="Path to local model")
    parser.add_argument("--max-num-batched-requests", default=4, type=int)
    parser.add_argument("--max-num-batched-tokens", default=8, type=int)
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--max-num-pages", default=16, type=int)
    parser.add_argument("--page-size", default=4096, type=int)
    parser.add_argument("--output-dir", default=None, help="Output directory for compiled artifacts")
    args = parser.parse_args()

    config = RunnerConfig(
        model=args.model,
        model_path=args.model_path,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_seq_length=args.max_seq_length,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        output_dir=args.output_dir,
    )
    app.state.runner_config = config
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
