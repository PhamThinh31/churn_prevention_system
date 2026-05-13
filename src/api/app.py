"""FastAPI app for the Churn Prevention System.

Online flow: customer_id → churn score → SASRec user vector → FAISS top-50
            → LGBM reranker → optional LLM rerank → decision layer → top-K with action.

Artifacts are loaded once at startup from data/features/ and data/processed/.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    HealthResponse,
    ProductRec,
    RecommendRequest,
    RecommendResponse,
)
from src.decision.retention import decide
from src.faiss.index import l2_normalize, load_index, topk
from src.models.retrieval.sasrec import SASRec

log = logging.getLogger("churn-api")
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "data" / "features"
PROCESSED = PROJECT_ROOT / "data" / "processed"


class State:
    def __init__(self):
        self.ready = False
        self.churn_scores: pd.DataFrame | None = None
        self.sasrec: SASRec | None = None
        self.vocab: dict | None = None
        self.faiss_index = None
        self.transactions: pd.DataFrame | None = None
        self.max_len: int = 50

    def load(self) -> dict[str, bool]:
        status = {}

        churn_path = ARTIFACTS / "churn_scores_gbm_test.parquet"
        if churn_path.exists():
            self.churn_scores = pd.read_parquet(churn_path).set_index("customer_id")
            status["churn_scores"] = True
        else:
            status["churn_scores"] = False

        sasrec_dir = ARTIFACTS / "sasrec"
        if (sasrec_dir / "sasrec.pt").exists() and (sasrec_dir / "vocab.pt").exists():
            vocab = torch.load(sasrec_dir / "vocab.pt")
            self.vocab = vocab
            vocab_size = len(vocab["id2item"]) + 2
            self.sasrec = SASRec(vocab_size=vocab_size, max_len=self.max_len)
            self.sasrec.load_state_dict(torch.load(sasrec_dir / "sasrec.pt", map_location="cpu"))
            self.sasrec.eval()
            status["sasrec"] = True
        else:
            status["sasrec"] = False

        faiss_path = ARTIFACTS / "item_index.faiss"
        if faiss_path.exists():
            self.faiss_index = load_index(faiss_path)
            status["faiss"] = True
        else:
            status["faiss"] = False

        tx_path = PROCESSED / "transactions_clean.parquet"
        if tx_path.exists():
            self.transactions = pd.read_parquet(tx_path)
            status["transactions"] = True
        else:
            status["transactions"] = False

        self.ready = all(status.values())
        return status


state = State()
app = FastAPI(title="Churn Prevention API", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    status = state.load()
    log.info("Artifact load: %s", status)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = {k: bool(v) for k, v in state.load().items()}
    return HealthResponse(status="ok" if state.ready else "degraded", components=status)


def _user_vector(customer_id: int) -> np.ndarray | None:
    if state.sasrec is None or state.transactions is None or state.vocab is None:
        return None
    hist = state.transactions[state.transactions["customer_id"] == customer_id]
    if hist.empty:
        return None
    items = hist.sort_values("invoice_date")["stock_code"].tolist()
    item2id = state.vocab["item2id"]
    enc = [item2id[i] for i in items if i in item2id]
    if not enc:
        return None
    enc = enc[-state.max_len:]
    padded = enc + [state.vocab["pad_id"]] * (state.max_len - len(enc))   # right-pad
    x = torch.tensor([padded], dtype=torch.long)
    with torch.no_grad():
        v = state.sasrec.user_vector(x).cpu().numpy()
    return l2_normalize(v)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    if not state.ready:
        raise HTTPException(503, "Artifacts not loaded — train models first.")

    if req.customer_id not in state.churn_scores.index:
        raise HTTPException(404, f"Unknown customer_id {req.customer_id}")

    p_churn = float(state.churn_scores.loc[req.customer_id, "p_churn_gbm"])

    uvec = _user_vector(req.customer_id)
    if uvec is None:
        raise HTTPException(404, "No purchase history for SASRec encoding")

    scores, ids = topk(state.faiss_index, uvec, k=req.top_k)
    id2item = state.vocab["id2item"]
    recs = [
        ProductRec(stock_code=str(id2item[int(i)]) if 0 <= int(i) < len(id2item) and id2item[int(i)] else str(int(i)),
                   score=float(s), rank=r)
        for r, (i, s) in enumerate(zip(ids[0], scores[0], strict=False))
    ]

    action = decide(req.customer_id, p_churn, [r.stock_code for r in recs])

    if req.use_llm_reranker and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from src.models.reranker.llm import LLMReranker
            hist_df = state.transactions[state.transactions["customer_id"] == req.customer_id].tail(30)
            cand_df = pd.DataFrame({
                "stock_code": [r.stock_code for r in recs],
                "description": [
                    state.transactions[state.transactions["stock_code"] == r.stock_code]["description"].head(1).iloc[0]
                    if (state.transactions["stock_code"] == r.stock_code).any() else r.stock_code
                    for r in recs
                ],
                "price": [
                    float(state.transactions[state.transactions["stock_code"] == r.stock_code]["price"].head(1).iloc[0])
                    if (state.transactions["stock_code"] == r.stock_code).any() else 0.0
                    for r in recs
                ],
                "score": [r.score for r in recs],
            })
            ordered = LLMReranker().rerank(hist_df, cand_df)
            recs_by_code = {r.stock_code: r for r in recs}
            recs = [recs_by_code[c] for c in ordered if c in recs_by_code]
            for i, r in enumerate(recs):
                r.rank = i
        except Exception as e:
            log.warning("LLM reranker failed; using LGBM order. err=%s", e)

    return RecommendResponse(
        customer_id=req.customer_id,
        p_churn=p_churn,
        risk_tier=action.risk_tier,
        suggested_discount_pct=action.suggested_discount_pct,
        top_products=recs,
    )
