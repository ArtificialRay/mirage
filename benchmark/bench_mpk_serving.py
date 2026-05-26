# SPDX-License-Identifier: Apache-2.0
"""Lightweight ShareGPT benchmark for the Mirage MPK online server.

Sends ShareGPT prompts at a configured RPS to a running
`mirage.engine.launch_server` instance and prints the same
"Serving Benchmark Result" table that `vllm bench serve` prints, by reusing
vllm's `async_request_openai_completions`, `get_request`, and
`calculate_metrics` primitives.

Usage::

    python -m mirage.engine.launch_server --model Qwen/Qwen3-0.6B --port 8000 &
    python mirage/benchmark/bench_mpk_serving.py \\
        --model Qwen/Qwen3-0.6B \\
        --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \\
        --num-prompts 50 --request-rate 2.0
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
from datetime import datetime
from typing import Any

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from vllm.benchmarks.datasets import ShareGPTDataset
from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    async_request_openai_completions,
)
from vllm.benchmarks.lib.ready_checker import wait_for_endpoint
from vllm.benchmarks.serve import calculate_metrics, get_request
from vllm.transformers_utils.tokenizer import get_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lightweight ShareGPT benchmark for Mirage MPK server.",
    )
    p.add_argument("--model", type=str, required=True,
                   help="Model name sent in the `model` field (server ignores it but it labels results).")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="HF tokenizer id; defaults to --model.")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dataset-path", type=str, required=True,
                   help="Path to ShareGPT_V3_unfiltered_cleaned_split.json")
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--request-rate", type=float, default=float("inf"),
                   help="Requests per second (Poisson by default). Use 'inf' to fire all at once.")
    p.add_argument("--burstiness", type=float, default=1.0,
                   help="Gamma shape parameter for arrival distribution. 1.0 == Poisson.")
    p.add_argument("--max-concurrency", type=int, default=None,
                   help="Cap on in-flight requests. Default: no cap.")
    p.add_argument("--sharegpt-output-len", type=int, default=None,
                   help="Fix expected output length per request. Default: use ShareGPT completion length.")
    p.add_argument("--ignore-eos", action="store_true", default=False,
                   help="Pass ignore_eos=true to the server (server may or may not honor it).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metric-percentiles", type=str, default="50,95,99")
    p.add_argument("--disable-tqdm", action="store_true", default=False)
    p.add_argument("--ready-timeout", type=int, default=600,
                   help="Seconds to wait for the server to become ready.")
    p.add_argument("--trust-remote-code", action="store_true", default=False)
    p.add_argument("--save-result", action="store_true", default=False)
    p.add_argument("--result-dir", type=str, default=None)
    p.add_argument("--result-filename", type=str, default=None)
    p.add_argument("--label", type=str, default=None,
                   help="Prefix for the saved result file. Default: 'mpk'.")
    return p.parse_args()


def _print_results(
    metrics,
    outputs,
    actual_output_lens,
    duration: float,
    request_rate: float,
    max_concurrency: int | None,
    selected_percentiles: list[float],
) -> dict[str, Any]:
    """Mirror the print block at vllm/benchmarks/serve.py:752 onward."""
    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failed))
    if max_concurrency is not None:
        print("{:<40} {:<10}".format("Maximum request concurrency:", max_concurrency))
    if request_rate != float("inf"):
        print("{:<40} {:<10.2f}".format("Request rate configured (RPS):", request_rate))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format(
        "Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format(
        "Output token throughput (tok/s):", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format(
        "Peak output token throughput (tok/s):", metrics.max_output_tokens_per_s))
    print("{:<40} {:<10.2f}".format(
        "Peak concurrent requests:", metrics.max_concurrent_requests))
    print("{:<40} {:<10.2f}".format(
        "Total Token throughput (tok/s):", metrics.total_token_throughput))

    result: dict[str, Any] = {
        "duration": duration,
        "completed": metrics.completed,
        "failed": metrics.failed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "max_output_tokens_per_s": metrics.max_output_tokens_per_s,
        "max_concurrent_requests": metrics.max_concurrent_requests,
        "input_lens": [o.prompt_len for o in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [o.ttft for o in outputs],
        "itls": [o.itl for o in outputs],
        "generated_texts": [o.generated_text for o in outputs],
        "errors": [o.error for o in outputs],
    }

    def emit(metric_attr: str, metric_label: str, header: str) -> None:
        print("{s:{c}^{n}}".format(s=header, n=50, c="-"))
        for stat in ("mean", "median"):
            v = getattr(metrics, f"{stat}_{metric_attr}_ms")
            print("{:<40} {:<10.2f}".format(f"{stat.capitalize()} {metric_label} (ms):", v))
            result[f"{stat}_{metric_attr}_ms"] = v
        result[f"std_{metric_attr}_ms"] = getattr(metrics, f"std_{metric_attr}_ms")
        for p, value in getattr(metrics, f"percentiles_{metric_attr}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_label} (ms):", value))
            result[f"p{p_word}_{metric_attr}_ms"] = value

    emit("ttft", "TTFT", "Time to First Token")
    emit("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    emit("itl", "ITL", "Inter-token Latency")
    emit("e2el", "E2EL", "End-to-end Latency")
    print("=" * 50)
    return result


async def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    tokenizer_id = args.tokenizer or args.model
    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)

    dataset = ShareGPTDataset(dataset_path=args.dataset_path, random_seed=args.seed)
    input_requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=args.num_prompts,
        output_len=args.sharegpt_output_len,
    )
    if not input_requests:
        raise RuntimeError("ShareGPT sampling returned 0 requests; check --dataset-path.")

    api_url = f"http://{args.host}:{args.port}/v1/completions"

    connector = aiohttp.TCPConnector(
        limit=args.max_concurrency or 0,
        limit_per_host=args.max_concurrency or 0,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
    )
    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    ) as session:
        # Ready check using the first sampled prompt.
        first = input_requests[0]
        test_input = RequestFuncInput(
            model=args.model,
            prompt=first.prompt,
            api_url=api_url,
            prompt_len=first.prompt_len,
            output_len=min(first.expected_output_len, 8),
            ignore_eos=args.ignore_eos,
        )
        test_out = await wait_for_endpoint(
            async_request_openai_completions,
            test_input,
            session,
            timeout_seconds=args.ready_timeout,
        )
        if not test_out.success:
            raise RuntimeError(
                f"Mirage MPK server at {api_url} not ready: {test_out.error}"
            )
        print(f"Server ready at {api_url}. Starting benchmark...")

        semaphore = (
            asyncio.Semaphore(args.max_concurrency)
            if args.max_concurrency
            else contextlib.nullcontext()
        )
        pbar = None if args.disable_tqdm else tqdm(total=len(input_requests))

        async def fire(req) -> Any:
            inp = RequestFuncInput(
                model=args.model,
                prompt=req.prompt,
                api_url=api_url,
                prompt_len=req.prompt_len,
                output_len=req.expected_output_len,
                ignore_eos=args.ignore_eos,
                request_id=req.request_id,
            )
            async with semaphore:
                return await async_request_openai_completions(
                    request_func_input=inp, session=session, pbar=pbar,
                )

        tasks: list[asyncio.Task] = []
        t0 = time.perf_counter()
        async for req, _ in get_request(
            input_requests,
            args.request_rate,
            burstiness=args.burstiness,
        ):
            tasks.append(asyncio.create_task(fire(req)))
        outputs = await asyncio.gather(*tasks)
        duration = time.perf_counter() - t0

        if pbar is not None:
            pbar.close()

    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]
    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=duration,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
        goodput_config_dict={},
    )

    result = _print_results(
        metrics, outputs, actual_output_lens,
        duration, args.request_rate, args.max_concurrency,
        selected_percentiles,
    )

    if args.save_result:
        meta = {
            "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "backend": "mpk",
            "model_id": args.model,
            "tokenizer_id": tokenizer_id,
            "num_prompts": args.num_prompts,
            "request_rate": (
                args.request_rate if args.request_rate < float("inf") else "inf"
            ),
            "burstiness": args.burstiness,
            "max_concurrency": args.max_concurrency,
        }
        result_json = {**meta, **result}

        label = args.label or "mpk"
        base_model = args.model.split("/")[-1]
        conc = f"-concurrency{args.max_concurrency}" if args.max_concurrency else ""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = args.result_filename or (
            f"{label}-{args.request_rate}qps{conc}-{base_model}-{ts}.json"
        )
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=2)
        print(f"Saved results to {file_name}")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
