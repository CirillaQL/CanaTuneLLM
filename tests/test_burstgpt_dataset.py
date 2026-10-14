"""scripts/make_burstgpt_dataset.py without the model's tokenizer: row filtering and
prompts of exactly the logged request length."""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_burstgpt_dataset as mk  # noqa: E402


class WordTokenizer:
    """One token per word, a BOS in front unless add_special_tokens=False; decoding a
    cut can glue its first two words (as subword tokenizers merge pieces at a cut),
    here when the cut has a multiple of 3 words and starts at an x word."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.words: list[str] = []

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]
        self.words += [w for w in text.split() if self.vocab[w] >= len(self.words)]
        return SimpleNamespace(input_ids=([-1] if add_special_tokens else []) + ids)

    def decode(self, ids) -> str:
        words = [self.words[i] for i in ids]
        if len(words) > 2 and len(words) % 3 == 0 and words[0].startswith("x"):
            words = [words[0] + words[1], *words[2:]]
        return " ".join(words)


def test_rows_are_filtered_by_model_and_length(tmp_path) -> None:
    path = tmp_path / "b.csv"
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(
            ["Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"]
        )
        w.writerows([
            [1, "ChatGPT", 100, 20, 120, "c"], [2, "GPT-4", 300, 0, 300, "c"],
            [3, "GPT-4", 4000, 200, 4200, "c"], [4, "GPT-4", 50, 10, 60, "c"],
        ])  # fmt: skip
    assert mk.load_rows([str(path)], "all", 4096) == ([(100, 20), (50, 10)], 4)
    assert mk.load_rows([str(path)], "GPT-4", 4096) == ([(50, 10)], 4)


def test_prompts_have_exactly_the_logged_length() -> None:
    tok = WordTokenizer()
    stream = mk.token_stream(tok, (f"w{i} x{i} y{i}" for i in range(10_000)), 5_000)
    data, misses = mk.build([(7, 3), (12, 40), (300, 1)], stream, tok, count=20, seed=1)
    assert len(data) == 20 and misses == [0] * 20
    for item in data:
        assert len(tok(item["prompt"]).input_ids) == item["request_tokens"]
        assert item["output_tokens"] in (3, 40, 1)


def test_a_window_keeps_every_row_in_time_order_with_its_arrival(tmp_path) -> None:
    path = tmp_path / "b.csv"
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(
            ["Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"]
        )
        w.writerows([
            [99, "ChatGPT", 10, 2, 12, "c"], [100, "ChatGPT", 12, 3, 15, "c"],
            [105, "GPT-4", 7, 0, 7, "c"], [103, "GPT-4", 9, 4, 13, "c"],
            [160, "ChatGPT", 5, 1, 6, "c"],
        ])  # fmt: skip
    rows, seen = mk.load_window([str(path)], "all", 4096, 100, 60)
    assert rows == [(0.0, 12, 3), (3.0, 9, 4)] and seen == 3  # 105: no response; 160: out
    tok = WordTokenizer()
    stream = mk.token_stream(tok, (f"w{i} x{i} y{i}" for i in range(10_000)), 5_000)
    data, misses = mk.build_window(rows, stream, tok, seed=1)
    assert [d["arrival_s"] for d in data] == [0.0, 3.0] and misses == [0, 0]
    assert [len(tok(d["prompt"]).input_ids) for d in data] == [12, 9]
