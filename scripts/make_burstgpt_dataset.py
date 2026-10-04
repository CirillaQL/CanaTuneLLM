#!/usr/bin/env python3
"""BurstGPT as a `vllm bench serve --dataset-name custom` dataset (jsonl: prompt, output_tokens).

BurstGPT (github.com/HPMLL/BurstGPT, CC-BY-4.0) logs the request and response token counts
of real ChatGPT / GPT-4 traffic, not its text. Each sampled row becomes a prompt of exactly
its request tokens, counted by the serving tokenizer as vllm bench and the proxy count it
(BOS included), cut from ShareGPT human turns at a row-specific offset (so prompts share no
prefix), and its response tokens become output_tokens (run with --ignore-eos and
--custom-output-len -1). vllm's own burstgpt dataset keeps only GPT-4 rows and decodes
synthetic token ids, which re-tokenize to other lengths.

Rows kept: the chosen model(s), request and response > 0, request + response <= --max-len.
The sample is seeded and already shuffled (use --disable-shuffle).

  make_burstgpt_dataset.py --csv BurstGPT_1.csv --sharegpt ShareGPT_V3.json \
      --tokenizer MODEL_PATH --out burstgpt.jsonl [--model all] [--rows 4000]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path


def load_rows(paths: Sequence[str], model: str, max_len: int) -> tuple[list[tuple[int, int]], int]:
    """-> (request, response) token counts of the kept rows, and the number of rows read."""
    rows, seen = [], 0
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                seen += 1
                if model != "all" and row["Model"] != model:
                    continue
                request, response = int(row["Request tokens"]), int(row["Response tokens"])
                if request > 0 and response > 0 and request + response <= max_len:
                    rows.append((request, response))
    return rows, seen


def sharegpt_texts(path: str) -> Iterable[str]:
    for item in json.loads(Path(path).read_text()):
        for turn in item.get("conversations", []):
            if turn.get("from") == "human" and turn.get("value", "").strip():
                yield turn["value"]


def token_stream(tokenizer, texts: Iterable[str], need: int) -> list[int]:
    """Token ids of the texts (no special tokens), separated by blank lines, >= need ids."""
    sep = tokenizer("\n\n", add_special_tokens=False).input_ids
    stream: list[int] = []
    for text in texts:
        stream += tokenizer(text, add_special_tokens=False).input_ids + sep
        if len(stream) >= need:
            return stream
    raise SystemExit(f"the ShareGPT text has {len(stream)} tokens, {need} needed")


def exact_prompt(tokenizer, stream: Sequence[int], offset: int, n: int) -> tuple[str, int]:
    """A prompt from stream[offset:] whose tokenized length (BOS included) is n, or the
    closest found -> (prompt, its length). Decoding and re-encoding can merge or split
    tokens at the cuts, so the end moves until the count matches; where it oscillates
    around n, the start moves by a token."""
    want = n - len(tokenizer("", add_special_tokens=True).input_ids)
    best = None
    for start in range(offset, offset + 8):
        m = max(1, want)
        for _ in range(6):
            text = tokenizer.decode(stream[start : start + m])
            got = len(tokenizer(text).input_ids)
            if best is None or abs(got - n) < abs(best[1] - n):
                best = (text, got)
            if got == n:
                return best
            m = max(1, m + n - got)
    return best


def build(rows, stream, tokenizer, count: int, seed: int) -> tuple[list[dict], list[int]]:
    rng = random.Random(seed)
    sample = rng.sample(rows, count) if count <= len(rows) else rng.choices(rows, k=count)
    out, misses = [], []
    for request, response in sample:
        offset = rng.randrange(0, len(stream) - 2 * request)
        prompt, got = exact_prompt(tokenizer, stream, offset, request)
        misses.append(got - request)
        out.append({"prompt": prompt, "output_tokens": response, "request_tokens": request})
    return out, misses


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", required=True, help="BurstGPT CSV file(s), comma separated")
    ap.add_argument("--sharegpt", required=True, help="ShareGPT json (prompt text)")
    ap.add_argument("--tokenizer", required=True, help="the served model's tokenizer")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="all", choices=("all", "ChatGPT", "GPT-4"))
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--max-len", type=int, default=4096, help="prompt + output limit")
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows, seen = load_rows(args.csv.split(","), args.model, args.max_len)
    if not rows:
        raise SystemExit("no BurstGPT rows kept")
    need = 4 * max(r for r, _ in rows) + 200_000
    stream = token_stream(tokenizer, sharegpt_texts(args.sharegpt), need)
    data, misses = build(rows, stream, tokenizer, args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for item in data:
            handle.write(json.dumps(item) + "\n")
    report = {
        "model": args.model,
        "rows_read": seen,
        "rows_kept": len(rows),
        "rows_written": len(data),
        "prompt_mean": round(statistics.fmean(d["request_tokens"] for d in data), 1),
        "output_mean": round(statistics.fmean(d["output_tokens"] for d in data), 1),
        "exact_prompts": sum(1 for x in misses if x == 0),
        "max_abs_length_error": max(abs(x) for x in misses),
    }
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
