from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from .models import RecommenderBase


def prediction_shift(
    clean_model: RecommenderBase,
    poisoned_model: RecommenderBase,
    users: Sequence[int],
    target_item: int,
    clean_sequences: Optional[Dict[int, List[int]]] = None,
    poisoned_sequences: Optional[Dict[int, List[int]]] = None,
) -> float:
    if not users:
        return 0.0
    users_arr = np.array(list(users), dtype=np.int64)
    items_arr = np.full_like(users_arr, int(target_item))
    clean_scores = clean_model.score(users_arr, items_arr, user_sequences=clean_sequences)
    poison_scores = poisoned_model.score(users_arr, items_arr, user_sequences=poisoned_sequences)
    return float((poison_scores - clean_scores).mean())


def hit_ndcg_at_k(
    model: RecommenderBase,
    users: Sequence[int],
    train_user_items: Dict[int, Set[int]],
    target_items: Set[int],
    num_items: int,
    k: int = 10,
    user_sequences: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, float]:
    if not users:
        return {"hit": 0.0, "ndcg": 0.0}

    all_items = np.arange(num_items, dtype=np.int64)
    hits = []
    ndcgs = []
    for user in users:
        seen = train_user_items.get(int(user), set())
        mask = np.ones(num_items, dtype=bool)
        if seen:
            seen_arr = np.fromiter(seen, dtype=np.int64)
            seen_arr = seen_arr[(seen_arr >= 0) & (seen_arr < num_items)]
            mask[seen_arr] = False
        candidates = all_items[mask]
        if len(candidates) == 0:
            continue
        users_arr = np.full(len(candidates), int(user), dtype=np.int64)
        scores = model.score(users_arr, candidates, user_sequences=user_sequences)
        top_idx = np.argpartition(-scores, kth=min(k, len(scores)) - 1)[:k]
        top_items = candidates[top_idx]
        # Stable rank to compute NDCG.
        order = np.argsort(-scores[top_idx])
        ranked = top_items[order]
        rel = np.array([1.0 if int(i) in target_items else 0.0 for i in ranked], dtype=np.float32)
        hit = 1.0 if rel.sum() > 0 else 0.0
        dcg = float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))
        ideal_rel = np.sort(rel)[::-1]
        idcg = float(np.sum(ideal_rel / np.log2(np.arange(2, len(ideal_rel) + 2))))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        hits.append(hit)
        ndcgs.append(ndcg)
    if not hits:
        return {"hit": 0.0, "ndcg": 0.0}
    return {"hit": float(np.mean(hits)), "ndcg": float(np.mean(ndcgs))}


def stealth_metrics(clean_train: pd.DataFrame, poisoned_train: pd.DataFrame) -> Dict[str, float]:
    if clean_train.empty or poisoned_train.empty:
        return {"rating_kl": 0.0, "profile_shift": 0.0, "interaction_ratio": 1.0}

    bins = np.array([0.0, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    clean_hist, _ = np.histogram(clean_train["rating"].to_numpy(), bins=bins, density=True)
    poison_hist, _ = np.histogram(poisoned_train["rating"].to_numpy(), bins=bins, density=True)
    eps = 1e-9
    clean_hist = np.clip(clean_hist, eps, None)
    poison_hist = np.clip(poison_hist, eps, None)
    rating_kl = float(np.sum(clean_hist * np.log(clean_hist / poison_hist)))
    # KL is theoretically non-negative; clip minor numeric underflow.
    rating_kl = float(max(0.0, rating_kl))

    clean_len = clean_train.groupby("user")["item"].count()
    poison_len = poisoned_train.groupby("user")["item"].count()
    shared_users = sorted(set(clean_len.index.tolist()) & set(poison_len.index.tolist()))
    if shared_users:
        profile_shift = float(
            np.abs(poison_len.loc[shared_users].to_numpy() - clean_len.loc[shared_users].to_numpy()).mean()
            / max(1.0, clean_len.loc[shared_users].to_numpy().mean())
        )
    else:
        profile_shift = 0.0
    interaction_ratio = float(len(poisoned_train) / max(1, len(clean_train)))
    return {"rating_kl": rating_kl, "profile_shift": profile_shift, "interaction_ratio": interaction_ratio}


def objective_from_intent(intent: str, shift: float) -> float:
    if intent.lower() == "push":
        return float(shift)
    return float(-shift)
