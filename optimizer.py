"""Mini Optimizer — candidate implementation.

Fill in the `optimize` function below. See README.md for the full spec.

You may add helper functions / classes / modules as needed, but the public
entry point `optimize(...)` must keep the exact signature and return shape
described in the README.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def optimize(
    trades_df: pd.DataFrame,
    stop_losses: list[float],
    take_profits: list[float],
    top_n: int = 5,
) -> list[dict]:
    """Return the top-N parameter combinations sorted by Sharpe (desc)."""

    # Edge cases
    if trades_df.empty or not stop_losses or not take_profits:
        return []

    # Drop rows with NaN in columns we use
    df = trades_df.dropna(subset=["pnl", "mae", "mfe"])

    if df.empty:
        return []

    # Extract numpy arrays once (fast)
    mae = df["mae"].to_numpy(dtype=float)
    mfe = df["mfe"].to_numpy(dtype=float)
    pnl = df["pnl"].to_numpy(dtype=float)
    n   = len(pnl)

    results = []

    for sl in stop_losses:
        for tp in take_profits:
            # Vectorised SL/TP logic — SL has priority over TP
            sl_hit = mae >= sl
            tp_hit = (~sl_hit) & (mfe >= tp)

            adj = np.where(sl_hit, -sl,
                  np.where(tp_hit,  tp, pnl))

            stopped_out = int(sl_hit.sum())
            took_profit = int(tp_hit.sum())
            total_pnl   = float(adj.sum())

            # Sharpe = mean / std (ddof=0)
            # Return 0.0 for single trade or zero std — never NaN/inf
            if n <= 1:
                sharpe = 0.0
            else:
                std = float(adj.std(ddof=0))
                sharpe = float(adj.mean() / std) if std > 0.0 else 0.0

            results.append({
                "stop_loss":   float(sl),
                "take_profit": float(tp),
                "sharpe":      sharpe,
                "total_pnl":   total_pnl,
                "stopped_out": stopped_out,
                "took_profit": took_profit,
            })

    # Deterministic sort:
    # 1. sharpe descending
    # 2. total_pnl descending
    # 3. (stop_loss, take_profit) ascending
    results.sort(
        key=lambda x: (-x["sharpe"], -x["total_pnl"], x["stop_loss"], x["take_profit"])
    )

    return results[:top_n]