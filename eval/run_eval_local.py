"""Run DeepEval prompt evaluations locally using Ollama as the base LLM.

This focuses on *base LLM* response quality (no retrieval eval):
- Uses a GoldContextRetriever to inject the gold `context` strings.
- Uses Ollama base model from config.yml for generation.
- Uses Gemini as the DeepEval judge model when GEMINI_API_KEY is provided.

Gold set: eval_dataset/911automate_gold_data_prompt_engg.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from src.rag.agent.agent import Agent
from src.rag.core.config import Config
from src.rag.core.types import RetrievedChunk


DEFAULT_GOLD_PATH = ROOT / "eval_dataset" / "911automate_gold_data_prompt_engg.json"


@dataclass(frozen=True)
class GoldExample:
    input: str
    expected_output: str
    context: list[str]
    source_doc: str
    question_type: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldExample":
        return cls(
            input=str(d.get("input") or ""),
            expected_output=str(d.get("expected_output") or ""),
            context=[str(x) for x in (d.get("context") or [])],
            source_doc=str(d.get("source_doc") or "none"),
            question_type=str(d.get("question_type") or "unknown"),
        )


class GoldContextRetriever:
    """Retriever that serves *only* the per-example gold context."""

    def __init__(self) -> None:
        self._ctx: list[str] = []
        self._source_doc: str = "unknown"

    def set_gold_context(self, context_strings: list[str], *, source_doc: str = "unknown") -> None:
        self._ctx = [c for c in context_strings if isinstance(c, str) and c.strip()]
        self._source_doc = source_doc

    def _to_retrieved_chunk(self, text: str, idx: int) -> RetrievedChunk:
        score = 0.99
        return RetrievedChunk(
            text=text,
            metadata={"doc_id": self._source_doc, "page": 0, "chunk_index": idx},
            score=score,
            provenance={
                "doc_id": self._source_doc,
                "page": 0,
                "chunk_index": idx,
                "source_path": "",
                "score": score,
            },
        )

    def retrieve(self, query: str, top_k: int | None = None, use_mmr: bool | None = None) -> list[RetrievedChunk]:
        return [self._to_retrieved_chunk(text, i) for i, text in enumerate(self._ctx)]


def _load_gold(path: Path) -> list[GoldExample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Gold set must be a JSON array of examples.")
    examples = [GoldExample.from_dict(x) for x in raw if isinstance(x, dict)]
    bad = [i for i, ex in enumerate(examples) if not ex.input or not ex.expected_output]
    if bad:
        raise ValueError(
            f"Gold set has {len(bad)} invalid examples (missing input/expected_output). First: {bad[0]}"
        )
    return examples


def _build_config(args) -> Config:
    cfg = Config.from_env(path=args.config)
    cfg.eval_mode = True
    return cfg


def _build_metrics() -> list:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required to run DeepEval judge metrics. "
            "Set GEMINI_API_KEY or run with --dry-run locally."
        )
    judge = GeminiModel(model="gemini-2.0-flash", api_key=api_key)
    return [
        AnswerRelevancyMetric(threshold=0.5, model=judge),
        FaithfulnessMetric(threshold=0.5, model=judge),
    ]


def _to_test_case(ex: GoldExample, *, actual_output: str) -> LLMTestCase:
    return LLMTestCase(
        input=ex.input,
        actual_output=actual_output,
        expected_output=ex.expected_output,
        context=ex.context,
        retrieval_context=ex.context,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DeepEval prompt evaluations locally (Ollama base LLM).")
    parser.add_argument("--gold", type=str, default=str(DEFAULT_GOLD_PATH), help="Path to gold JSON dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config.yml"),
        help="Path to config.yml (default: repo root config.yml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset and build testcases without calling any external LLMs.",
    )
    args = parser.parse_args(argv)

    gold_path = Path(args.gold)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold dataset not found: {gold_path}")

    examples = _load_gold(gold_path)
    config = _build_config(args)

    gold_retriever = GoldContextRetriever()
    agent = Agent(retriever=gold_retriever, config=config)

    test_cases: list[LLMTestCase] = []
    for ex in examples:
        gold_retriever.set_gold_context(ex.context, source_doc=ex.source_doc)
        if args.dry_run:
            actual = "(dry-run: skipped LLM call)"
        else:
            result = agent.handle(ex.input)
            actual = str(result.get("response") or "")
        test_cases.append(_to_test_case(ex, actual_output=actual))

    if args.dry_run:
        print(f"DRY RUN OK: loaded {len(test_cases)} test cases from {gold_path}")
        return 0

    metrics = _build_metrics()
    evaluate(test_cases, metrics=metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

