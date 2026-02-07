from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    time_col: str
    id_col: str
    target_col: str
    failure_event_col: str
    lead_time_hours: int
    feature_cols: Optional[List[str]] = None
    drop_cols: Optional[List[str]] = None


def _ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df


def _infer_rows_per_hour(df: pd.DataFrame, time_col: str, id_col: str) -> int:
    # Infer sampling frequency from median time delta per machine
    deltas = []
    for _, g in df.groupby(id_col):
        if g.shape[0] < 3:
            continue
        t = g[time_col].sort_values()
        dt = t.diff().dropna().dt.total_seconds() / 60.0
        if not dt.empty:
            deltas.append(dt.median())
    if not deltas:
        return 60
    median_minutes = float(np.median(deltas))
    if median_minutes <= 0:
        return 60
    return max(1, int(round(60.0 / median_minutes)))


def make_target_from_failure(
    df: pd.DataFrame,
    time_col: str,
    id_col: str,
    failure_event_col: str,
    target_col: str,
    lead_time_hours: int,
) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_datetime(df, time_col)

    def _label_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col)
        g = g.set_index(time_col)
        # Reverse time to convert look-ahead into look-back
        rev = g.iloc[::-1]
        rev_shift = rev[failure_event_col].shift(1)
        future = rev_shift.rolling(f"{lead_time_hours}H", min_periods=1).max()
        g[target_col] = future.iloc[::-1].fillna(0).astype(int)
        g = g.reset_index()
        return g

    df = df.groupby(id_col, group_keys=False).apply(_label_group)
    return df


def clean_and_interpolate(df: pd.DataFrame, time_col: str, id_col: str) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_datetime(df, time_col)
    df = df.sort_values([id_col, time_col])

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    def _interp(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col)
        g[num_cols] = g[num_cols].interpolate(method="linear", limit_direction="both")
        g[num_cols] = g[num_cols].fillna(method="ffill").fillna(method="bfill")
        return g

    return df.groupby(id_col, group_keys=False).apply(_interp)


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df = _ensure_datetime(df, cfg.time_col)

    # Determine base numeric features
    drop_cols = set([cfg.time_col, cfg.id_col, cfg.target_col, cfg.failure_event_col])
    if cfg.drop_cols:
        drop_cols.update(cfg.drop_cols)

    if cfg.feature_cols:
        base_features = list(cfg.feature_cols)
    else:
        base_features = [
            c
            for c in df.select_dtypes(include=["number"]).columns
            if c not in drop_cols
        ]

    rows_per_hour = _infer_rows_per_hour(df, cfg.time_col, cfg.id_col)
    win_1h = max(1, rows_per_hour * 1)
    win_4h = max(1, rows_per_hour * 4)
    win_8h = max(1, rows_per_hour * 8)

    def _fe_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(cfg.time_col).copy()
        for col in base_features:
            g[f"{col}_lag1"] = g[col].shift(1)
            g[f"{col}_lag2"] = g[col].shift(2)

            # Rolling stats (shifted to avoid leakage)
            g[f"{col}_roll1h_mean"] = g[col].rolling(win_1h).mean().shift(1)
            g[f"{col}_roll4h_mean"] = g[col].rolling(win_4h).mean().shift(1)
            g[f"{col}_roll8h_mean"] = g[col].rolling(win_8h).mean().shift(1)
            g[f"{col}_roll4h_std"] = g[col].rolling(win_4h).std().shift(1)

            # Exponential moving averages (shifted)
            g[f"{col}_ema4h"] = g[col].ewm(span=win_4h, adjust=False).mean().shift(1)

        return g

    df = df.groupby(cfg.id_col, group_keys=False).apply(_fe_group)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df, feature_cols
