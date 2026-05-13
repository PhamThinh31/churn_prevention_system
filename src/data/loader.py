"""Load and clean Online Retail II transactions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["InvoiceDate"], dtype={"Customer ID": "Float64"})
    return df.rename(columns=RAW_COLUMNS)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows we cannot use for churn / recsys labeling.

    Rules:
      - require customer_id
      - drop returns (quantity <= 0) and zero/negative price
      - drop non-product stock codes (POSTAGE, DOT, M, BANK CHARGES, etc.)
      - drop exact duplicates
    """
    out = df.dropna(subset=["customer_id"]).copy()
    out["customer_id"] = out["customer_id"].astype("int64")

    out = out[(out["quantity"] > 0) & (out["price"] > 0)]

    # Heuristic: real product codes are mostly numeric prefix.
    # Drop the well-known non-product codes.
    non_product = {"POST", "DOT", "M", "BANK CHARGES", "AMAZONFEE", "C2", "TEST", "ADJUST"}
    out = out[~out["stock_code"].astype(str).str.upper().isin(non_product)]

    out = out.drop_duplicates()
    out["revenue"] = out["quantity"] * out["price"]
    return out.reset_index(drop=True)
