from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class PreparedData:
    interactions: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    num_users: int
    num_items: int
    user_id_to_index: Dict[str, int]
    item_id_to_index: Dict[str, int]
    index_to_user_id: Dict[int, str]
    index_to_item_id: Dict[int, str]
    item_category: Dict[int, str]
    category_to_items: Dict[str, List[int]]


def _normalize_id_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _build_mapping(values: pd.Series) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniq = sorted(values.unique().tolist())
    idx = {v: i for i, v in enumerate(uniq)}
    rev = {i: v for v, i in idx.items()}
    return idx, rev


def _split_leave_two_out(
    df: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows = []
    val_rows = []
    test_rows = []
    rng = np.random.default_rng(seed)
    for _, g in df.groupby("user"):
        if "timestamp" in g.columns and g["timestamp"].notna().any():
            g_sorted = g.sort_values("timestamp")
        else:
            # Keep a deterministic pseudo-order when timestamps are missing.
            g_sorted = g.sample(frac=1.0, random_state=int(rng.integers(1, 1_000_000)))
        if len(g_sorted) >= 3:
            train_rows.append(g_sorted.iloc[:-2])
            val_rows.append(g_sorted.iloc[[-2]])
            test_rows.append(g_sorted.iloc[[-1]])
        elif len(g_sorted) == 2:
            train_rows.append(g_sorted.iloc[[0]])
            val_rows.append(g_sorted.iloc[[1]])
        elif len(g_sorted) == 1:
            train_rows.append(g_sorted.iloc[[0]])
    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=df.columns)
    return train_df, val_df, test_df


def prepare_data(
    interactions_path: str,
    items_path: str,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    timestamp_col: Optional[str] = "timestamp",
    category_col: str = "category",
    min_positive_rating: float = 4.0,
    seed: int = 42,
) -> PreparedData:
    inter = pd.read_csv(interactions_path)
    item_meta = pd.read_csv(items_path)

    required_inter_cols = {user_col, item_col, rating_col}
    if not required_inter_cols.issubset(inter.columns):
        raise ValueError(f"interactions.csv missing required columns: {required_inter_cols}")
    required_item_cols = {item_col, category_col}
    if not required_item_cols.issubset(item_meta.columns):
        raise ValueError(f"items.csv missing required columns: {required_item_cols}")

    inter = inter.copy()
    inter[user_col] = _normalize_id_column(inter[user_col])
    inter[item_col] = _normalize_id_column(inter[item_col])
    inter[rating_col] = inter[rating_col].astype(float)
    if timestamp_col and timestamp_col in inter.columns:
        inter["timestamp"] = pd.to_numeric(inter[timestamp_col], errors="coerce")
    else:
        inter["timestamp"] = np.nan

    # Convert to implicit-positive space used by the paper settings.
    inter = inter[inter[rating_col] >= min_positive_rating].copy()
    if inter.empty:
        raise ValueError("No positive interactions left after rating thresholding.")

    # De-duplicate repeated (user, item) logs to avoid train/val/test leakage on
    # the same interaction pair when users re-rate the same item multiple times.
    inter["_orig_order"] = np.arange(len(inter), dtype=np.int64)
    if inter["timestamp"].notna().any():
        inter = inter.sort_values([user_col, item_col, "timestamp", "_orig_order"], kind="mergesort")
    else:
        inter = inter.sort_values([user_col, item_col, "_orig_order"], kind="mergesort")
    inter = inter.drop_duplicates(subset=[user_col, item_col], keep="last").drop(columns=["_orig_order"])

    user_map, rev_user_map = _build_mapping(inter[user_col])
    item_map, rev_item_map = _build_mapping(inter[item_col])

    inter["user"] = inter[user_col].map(user_map).astype(int)
    inter["item"] = inter[item_col].map(item_map).astype(int)
    inter["rating"] = inter[rating_col].astype(float)
    inter["is_malicious"] = 0
    inter = inter[["user", "item", "rating", "timestamp", "is_malicious"]]

    item_meta = item_meta.copy()
    item_meta[item_col] = _normalize_id_column(item_meta[item_col])
    item_meta = item_meta[item_meta[item_col].isin(item_map)].copy()
    item_meta["item"] = item_meta[item_col].map(item_map).astype(int)
    item_meta["category"] = item_meta[category_col].astype(str).str.strip()

    item_category = {int(r.item): r.category for r in item_meta[["item", "category"]].itertuples(index=False)}
    # Fill unknown categories to avoid key errors.
    for item_idx in range(len(item_map)):
        item_category.setdefault(item_idx, "UNKNOWN")
    category_to_items: Dict[str, List[int]] = {}
    for item_idx, cat in item_category.items():
        category_to_items.setdefault(cat, []).append(item_idx)

    train, val, test = _split_leave_two_out(inter, seed=seed)

    return PreparedData(
        interactions=inter,
        train=train,
        val=val,
        test=test,
        num_users=len(user_map),
        num_items=len(item_map),
        user_id_to_index=user_map,
        item_id_to_index=item_map,
        index_to_user_id=rev_user_map,
        index_to_item_id=rev_item_map,
        item_category=item_category,
        category_to_items=category_to_items,
    )


def build_user_item_sets(df: pd.DataFrame) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    if df.empty:
        return out
    for user, g in df.groupby("user"):
        out[int(user)] = set(g["item"].astype(int).tolist())
    return out


def build_user_sequences(df: pd.DataFrame, max_len: int) -> Dict[int, List[int]]:
    seqs: Dict[int, List[int]] = {}
    if df.empty:
        return seqs
    for user, g in df.groupby("user"):
        g_sorted = g.sort_values("timestamp")
        seq = g_sorted["item"].astype(int).tolist()
        seqs[int(user)] = seq[-max_len:]
    return seqs


def append_malicious_rows(clean_train: pd.DataFrame, malicious_rows: pd.DataFrame) -> pd.DataFrame:
    if malicious_rows.empty:
        return clean_train.copy()
    out = pd.concat([clean_train, malicious_rows], ignore_index=True)
    out["user"] = out["user"].astype(int)
    out["item"] = out["item"].astype(int)
    out["rating"] = out["rating"].astype(float)
    out["is_malicious"] = out["is_malicious"].astype(int)
    return out


def next_fake_user_id(train_df: pd.DataFrame, num_users: int) -> int:
    if train_df.empty:
        return num_users
    return max(num_users, int(train_df["user"].max()) + 1)
