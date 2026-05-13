"""LLM-based reranker.

Given a customer's purchase history + the top-K candidate products, ask the LLM
to reorder candidates to maximize the likelihood the customer would buy them.

This is a thin, swappable client. The training-quality ranker (LGBMRanker)
remains the source of truth — this is a final pass for the top-20 only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd

SYSTEM = (
    "You are a retail recommendation reranker. Given a customer's past purchases "
    "and a candidate set of products with metadata, reorder the candidates to maximize "
    "the probability the customer will purchase. Prioritize relevance to past behavior. "
    "Return ONLY a JSON array of stock_codes in the new order, no commentary."
)


def _format_history(history: pd.DataFrame, max_items: int = 30) -> str:
    h = history.tail(max_items).copy()
    return "\n".join(f"- {row.stock_code}: {row.description}" for row in h.itertuples())


def _format_candidates(candidates: pd.DataFrame) -> str:
    return "\n".join(
        f"- {row.stock_code}: {row.description} (price={row.price:.2f}, score={row.score:.3f})"
        for row in candidates.itertuples()
    )


@dataclass
class LLMRerankerConfig:
    # Default Anthropic model. Override to use a different snapshot
    # (e.g. via the ANTHROPIC_MODEL env var read by the caller).
    model: str = "claude-opus-4-5"
    max_tokens: int = 1024
    temperature: float = 0.0
    top_k_to_rerank: int = 20


class LLMReranker:
    def __init__(self, config: LLMRerankerConfig | None = None):
        self.config = config or LLMRerankerConfig()
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def rerank(self, history: pd.DataFrame, candidates: pd.DataFrame) -> list[str]:
        cands = candidates.head(self.config.top_k_to_rerank)
        user_msg = (
            f"Customer's recent purchases:\n{_format_history(history)}\n\n"
            f"Candidate products to reorder:\n{_format_candidates(cands)}\n\n"
            "Return a JSON array of stock_codes in the optimal order."
        )
        resp = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()
        # be lenient — strip code fences if present
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            ordered = json.loads(text)
        except json.JSONDecodeError:
            # fallback: keep original order
            return cands["stock_code"].tolist()
        return [c for c in ordered if c in set(cands["stock_code"].astype(str))]
