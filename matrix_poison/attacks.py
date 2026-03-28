from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import next_fake_user_id


@dataclass
class AttackRequest:
    epsilon: float
    intent: str  # push | nuke
    target_item: int
    target_items: List[int]
    target_category_items: List[int]
    max_profile_len: int = 30
    seed: int = 42


def _non_target_pool(num_items: int, target_category_items: List[int]) -> List[int]:
    tgt = set(int(x) for x in target_category_items)
    return [i for i in range(num_items) if i not in tgt]


def _profile_len(train_df: pd.DataFrame, max_profile_len: int) -> int:
    if train_df.empty:
        return min(20, max_profile_len)
    avg_len = int(train_df.groupby("user")["item"].count().mean())
    return int(np.clip(avg_len, 8, max_profile_len))


def _popular_items(train_df: pd.DataFrame, top_n: int) -> List[int]:
    if train_df.empty:
        return []
    return train_df["item"].value_counts().head(top_n).index.astype(int).tolist()


def _item_avg_rating(train_df: pd.DataFrame) -> Dict[int, float]:
    if train_df.empty:
        return {}
    return (
        train_df.groupby("item")["rating"]
        .mean()
        .astype(float)
        .to_dict()
    )


def _nuke_pools(
    train_df: pd.DataFrame,
    num_items: int,
    target_category_items: List[int],
    target_items: List[int],
) -> Tuple[List[int], List[int], List[int]]:
    target_set = set(int(x) for x in target_items)
    category_pool = [int(x) for x in target_category_items]
    category_non_target = [i for i in category_pool if i not in target_set]
    non_target = _non_target_pool(num_items, category_pool)
    if not non_target:
        non_target = list(range(num_items))
    popular = _popular_items(train_df, top_n=min(300, num_items))
    decoy = [i for i in popular if i not in target_set]
    if not decoy:
        decoy = non_target
    return decoy, category_non_target, non_target


def _sample_without_replacement(
    rng: np.random.Generator,
    pool: List[int],
    k: int,
    probs: Optional[np.ndarray] = None,
) -> List[int]:
    if k <= 0 or not pool:
        return []
    k = min(int(k), len(pool))
    arr = np.array(pool, dtype=np.int64)
    if probs is not None:
        p = np.clip(probs.astype(np.float64), 0.0, None)
        if p.sum() <= 0:
            p = np.full(len(arr), 1.0 / len(arr), dtype=np.float64)
        else:
            p = p / p.sum()
        return rng.choice(arr, size=k, replace=False, p=p).astype(int).tolist()
    return rng.choice(arr, size=k, replace=False).astype(int).tolist()


def _push_profile_len(train_df: pd.DataFrame, max_profile_len: int) -> int:
    upper = max(8, int(max_profile_len))
    lower = min(12, upper)
    cap = max(lower, min(24, upper))
    base = _profile_len(train_df, upper)
    return int(np.clip(base, lower, cap))


def _push_pools(
    train_df: pd.DataFrame,
    num_items: int,
    target_item: int,
    target_category_items: List[int],
) -> Tuple[List[int], List[int]]:
    category_set = set(int(x) for x in target_category_items)
    popular = _popular_items(train_df, top_n=min(max(200, num_items // 4), num_items))
    if not popular:
        popular = list(range(num_items))
    item_avg = _item_avg_rating(train_df)
    anchors = [i for i in popular if i in category_set and int(i) != int(target_item)][:3]
    filler = [
        i for i in popular
        if i not in category_set and int(i) != int(target_item) and item_avg.get(int(i), 4.0) >= 3.6
    ]
    if len(filler) < 30:
        filler = [i for i in popular if i not in category_set and int(i) != int(target_item)]
    if len(filler) < 10:
        filler = [i for i in range(num_items) if i not in category_set and i != int(target_item)]
    if not filler:
        filler = [i for i in range(num_items) if i != int(target_item)]
    return anchors, filler


def _rows_for_push_users(
    users: List[int],
    item_lists: List[List[int]],
    target_item: int,
    rng: np.random.Generator,
    base_rating: float,
) -> List[dict]:
    rows: List[dict] = []
    target_item = int(target_item)
    for u, items in zip(users, item_lists):
        ordered_items = [int(i) for i in items if int(i) != target_item]
        if target_item in set(int(i) for i in items):
            ordered_items.append(target_item)
        base_ts = 1_700_000_000 + int(u) * 1_000
        for offset, i in enumerate(ordered_items):
            if int(i) == target_item:
                rating = float(np.clip(rng.normal(4.95, 0.08), 4.6, 5.0))
            else:
                rating = float(np.clip(rng.normal(base_rating, 0.2), 4.0, 5.0))
            rows.append(
                {
                    "user": int(u),
                    "item": int(i),
                    "rating": rating,
                    "timestamp": float(base_ts + offset),
                    "is_malicious": 1,
                }
            )
    return rows


def _rows_for_users(
    users: List[int],
    item_lists: List[List[int]],
    base_rating: float,
    rng: np.random.Generator,
    low_items: Optional[set] = None,
    low_rating: float = 1.2,
) -> List[dict]:
    rows: List[dict] = []
    low_items = low_items or set()
    for u, items in zip(users, item_lists):
        for i in items:
            if int(i) in low_items:
                rating = float(np.clip(rng.normal(low_rating, 0.2), 1.0, 2.0))
            else:
                rating = float(np.clip(rng.normal(base_rating, 0.3), 1.0, 5.0))
            rows.append(
                {
                    "user": int(u),
                    "item": int(i),
                    "rating": rating,
                    "timestamp": np.nan,
                    "is_malicious": 1,
                }
            )
    return rows


def random_attack(
    train_df: pd.DataFrame,
    start_user_id: int,
    num_fake_users: int,
    num_items: int,
    req: AttackRequest,
) -> pd.DataFrame:
    rng = np.random.default_rng(req.seed + 11)
    profile_len = _push_profile_len(train_df, req.max_profile_len) if req.intent == "push" else _profile_len(train_df, req.max_profile_len)
    category_pool = req.target_category_items if req.target_category_items else list(range(num_items))
    decoy_pool, category_non_target, non_target = _nuke_pools(
        train_df=train_df,
        num_items=num_items,
        target_category_items=category_pool,
        target_items=req.target_items,
    )
    push_anchors, push_filler = _push_pools(
        train_df=train_df,
        num_items=num_items,
        target_item=req.target_item,
        target_category_items=category_pool,
    )
    users = list(range(start_user_id, start_user_id + num_fake_users))
    items_per_user: List[List[int]] = []
    target_set = set(int(x) for x in req.target_items)
    for _ in users:
        base_pool = push_filler if req.intent == "push" else decoy_pool
        chosen = set(_sample_without_replacement(rng, base_pool, profile_len))
        if req.intent == "push":
            chosen.add(int(req.target_item))
            chosen.update(_sample_without_replacement(rng, push_anchors, 1))
            filler_need = max(0, profile_len - len(chosen))
            chosen.update(_sample_without_replacement(rng, push_filler, filler_need))
        else:
            # Nuke: suppress head target items by shifting attention to category alternatives + outside popular items.
            if category_non_target:
                chosen.update(rng.choice(category_non_target, size=min(6, len(category_non_target)), replace=False).astype(int).tolist())
            chosen.update(rng.choice(decoy_pool, size=min(10, len(decoy_pool)), replace=False).astype(int).tolist())
            chosen.update(rng.choice(non_target, size=min(4, len(non_target)), replace=False).astype(int).tolist())
            if req.target_items:
                chosen.update([int(x) for x in req.target_items])
        items_per_user.append(list(chosen))
    if req.intent == "push":
        rows = _rows_for_push_users(
            users=users,
            item_lists=items_per_user,
            target_item=req.target_item,
            rng=rng,
            base_rating=4.6,
        )
    else:
        rows = _rows_for_users(
            users,
            items_per_user,
            base_rating=4.5,
            rng=rng,
            low_items=target_set if req.intent == "nuke" else None,
        )
    return pd.DataFrame(rows)


def average_attack(
    train_df: pd.DataFrame,
    start_user_id: int,
    num_fake_users: int,
    num_items: int,
    req: AttackRequest,
) -> pd.DataFrame:
    rng = np.random.default_rng(req.seed + 17)
    profile_len = _push_profile_len(train_df, req.max_profile_len) if req.intent == "push" else _profile_len(train_df, req.max_profile_len)
    item_avg = _item_avg_rating(train_df)
    category_pool = req.target_category_items if req.target_category_items else list(range(num_items))
    decoy_pool, category_non_target, non_target = _nuke_pools(
        train_df=train_df,
        num_items=num_items,
        target_category_items=category_pool,
        target_items=req.target_items,
    )
    push_anchors, push_filler = _push_pools(
        train_df=train_df,
        num_items=num_items,
        target_item=req.target_item,
        target_category_items=category_pool,
    )
    all_items = np.arange(num_items, dtype=np.int64)
    if req.intent == "push":
        sample_items = np.array(push_filler, dtype=np.int64)
    else:
        mix_pool = list(dict.fromkeys(decoy_pool + category_non_target + non_target))
        sample_items = np.array(mix_pool, dtype=np.int64)
    weights = np.array([item_avg.get(int(i), 4.0) for i in sample_items], dtype=np.float64)
    weights = np.clip(weights, 0.1, None)
    weights = weights / weights.sum()

    users = list(range(start_user_id, start_user_id + num_fake_users))
    target_set = set(int(x) for x in req.target_items)
    if req.intent == "push":
        items_per_user: List[List[int]] = []
        for _ in users:
            chosen = set()
            chosen.add(int(req.target_item))
            chosen.update(_sample_without_replacement(rng, push_anchors, 1))
            filler_need = max(0, profile_len - len(chosen))
            chosen.update(
                _sample_without_replacement(
                    rng,
                    sample_items.astype(int).tolist(),
                    filler_need,
                    probs=weights,
                )
            )
            filler_need = max(0, profile_len - len(chosen))
            chosen.update(_sample_without_replacement(rng, push_filler, filler_need))
            items_per_user.append(list(chosen))
        return pd.DataFrame(
            _rows_for_push_users(
                users=users,
                item_lists=items_per_user,
                target_item=req.target_item,
                rng=rng,
                base_rating=4.55,
            )
        )

    rows: List[dict] = []
    for u in users:
        chosen = set(
            rng.choice(sample_items, size=min(profile_len, len(sample_items)), replace=False, p=weights).astype(int).tolist()
        )
        if category_non_target:
            chosen.update(rng.choice(category_non_target, size=min(6, len(category_non_target)), replace=False).astype(int).tolist())
        chosen.update(rng.choice(decoy_pool, size=min(12, len(decoy_pool)), replace=False).astype(int).tolist())
        chosen.update(rng.choice(non_target, size=min(4, len(non_target)), replace=False).astype(int).tolist())
        if req.target_items:
            chosen.update([int(x) for x in req.target_items])
        for i in chosen:
            if int(i) in target_set:
                base = 1.2
                rating = float(np.clip(rng.normal(base, 0.25), 1.0, 2.0))
            else:
                base = item_avg.get(int(i), 4.0)
                rating = float(np.clip(rng.normal(base, 0.25), 1.0, 5.0))
            rows.append({"user": int(u), "item": int(i), "rating": rating, "timestamp": np.nan, "is_malicious": 1})
    return pd.DataFrame(rows)


def bandwagon_attack(
    train_df: pd.DataFrame,
    start_user_id: int,
    num_fake_users: int,
    num_items: int,
    req: AttackRequest,
) -> pd.DataFrame:
    rng = np.random.default_rng(req.seed + 23)
    profile_len = _push_profile_len(train_df, req.max_profile_len) if req.intent == "push" else _profile_len(train_df, req.max_profile_len)
    popular = _popular_items(train_df, top_n=min(200, num_items))
    if not popular:
        popular = list(range(min(100, num_items)))
    category_pool = req.target_category_items if req.target_category_items else list(range(num_items))
    decoy_pool, category_non_target, non_target = _nuke_pools(
        train_df=train_df,
        num_items=num_items,
        target_category_items=category_pool,
        target_items=req.target_items,
    )
    popular_non_target = [i for i in popular if i in set(non_target)]
    if not popular_non_target:
        popular_non_target = non_target
    push_anchors, push_filler = _push_pools(
        train_df=train_df,
        num_items=num_items,
        target_item=req.target_item,
        target_category_items=category_pool,
    )
    push_hot = [i for i in popular_non_target if i in set(push_filler)]
    if len(push_hot) < 10:
        push_hot = push_filler

    users = list(range(start_user_id, start_user_id + num_fake_users))
    items_per_user: List[List[int]] = []
    target_set = set(int(x) for x in req.target_items)
    for _ in users:
        base_pool = popular if req.intent == "push" else decoy_pool
        chosen = set(rng.choice(base_pool, size=min(max(3, profile_len // 2), len(base_pool)), replace=False).astype(int).tolist())
        if req.intent == "push":
            chosen = {int(req.target_item)}
            chosen.update(_sample_without_replacement(rng, push_anchors, 1))
            chosen.update(_sample_without_replacement(rng, push_hot, max(3, profile_len // 2)))
            filler_need = max(0, profile_len - len(chosen))
            chosen.update(_sample_without_replacement(rng, push_filler, filler_need))
        else:
            if category_non_target:
                chosen.update(
                    rng.choice(category_non_target, size=min(6, len(category_non_target)), replace=False).astype(int).tolist()
                )
            chosen.update(rng.choice(decoy_pool, size=min(14, len(decoy_pool)), replace=False).astype(int).tolist())
            if req.target_items:
                chosen.update([int(x) for x in req.target_items])
        fill_pool = push_filler if req.intent == "push" else base_pool
        while len(chosen) < min(profile_len, len(fill_pool)):
            chosen.add(int(fill_pool[int(rng.integers(0, len(fill_pool)))]))
        items_per_user.append(list(chosen))
    if req.intent == "push":
        rows = _rows_for_push_users(
            users=users,
            item_lists=items_per_user,
            target_item=req.target_item,
            rng=rng,
            base_rating=4.75,
        )
    else:
        rows = _rows_for_users(
            users,
            items_per_user,
            base_rating=4.6,
            rng=rng,
            low_items=target_set if req.intent == "nuke" else None,
        )
    return pd.DataFrame(rows)


def segmented_attack(
    train_df: pd.DataFrame,
    start_user_id: int,
    num_fake_users: int,
    num_items: int,
    req: AttackRequest,
) -> pd.DataFrame:
    rng = np.random.default_rng(req.seed + 31)
    profile_len = _push_profile_len(train_df, req.max_profile_len) if req.intent == "push" else _profile_len(train_df, req.max_profile_len)
    category_pool = req.target_category_items if req.target_category_items else list(range(num_items))
    decoy_pool, category_non_target, _ = _nuke_pools(
        train_df=train_df,
        num_items=num_items,
        target_category_items=category_pool,
        target_items=req.target_items,
    )
    category_pool = [int(x) for x in category_pool]
    non_target_pool = _non_target_pool(num_items, category_pool)
    if not non_target_pool:
        non_target_pool = list(range(num_items))
    push_anchors, push_filler = _push_pools(
        train_df=train_df,
        num_items=num_items,
        target_item=req.target_item,
        target_category_items=category_pool,
    )

    users = list(range(start_user_id, start_user_id + num_fake_users))
    if req.intent == "push":
        items_per_user: List[List[int]] = []
        for idx, _ in enumerate(users):
            focused = (idx % 2 == 0)
            chosen = set()
            chosen.add(int(req.target_item))
            anchor_k = 2 if focused else 1
            chosen.update(_sample_without_replacement(rng, push_anchors, anchor_k))
            base_push_pool = push_filler if not focused else list(dict.fromkeys(push_anchors + push_filler))
            filler_need = max(0, profile_len - len(chosen))
            chosen.update(_sample_without_replacement(rng, base_push_pool, filler_need))
            items_per_user.append(list(chosen))
        return pd.DataFrame(
            _rows_for_push_users(
                users=users,
                item_lists=items_per_user,
                target_item=req.target_item,
                rng=rng,
                base_rating=4.6,
            )
        )

    rows: List[dict] = []
    target_set = set(int(x) for x in req.target_items)
    for idx, u in enumerate(users):
        focused = (idx % 2 == 0)
        chosen = set()
        if focused:
            chosen.update(rng.choice(category_pool, size=min(max(4, profile_len // 2), len(category_pool)), replace=False).astype(int).tolist())
        else:
            chosen.update(rng.choice(non_target_pool, size=min(max(4, profile_len // 2), len(non_target_pool)), replace=False).astype(int).tolist())

        if category_non_target:
            chosen.update(rng.choice(category_non_target, size=min(6, len(category_non_target)), replace=False).astype(int).tolist())
        chosen.update(rng.choice(decoy_pool, size=min(12, len(decoy_pool)), replace=False).astype(int).tolist())
        chosen.update(rng.choice(non_target_pool, size=min(4, len(non_target_pool)), replace=False).astype(int).tolist())
        if req.target_items:
            chosen.update([int(x) for x in req.target_items])

        base_pool = category_pool if focused else non_target_pool
        while len(chosen) < min(profile_len, len(base_pool)):
            chosen.add(int(base_pool[int(rng.integers(0, len(base_pool)))]))

        base = 4.7 if focused else 4.3
        for i in chosen:
            if int(i) in target_set:
                rating = float(np.clip(rng.normal(1.2, 0.2), 1.0, 2.0))
            else:
                rating = float(np.clip(rng.normal(base, 0.35), 1.0, 5.0))
            rows.append({"user": int(u), "item": int(i), "rating": rating, "timestamp": np.nan, "is_malicious": 1})
    return pd.DataFrame(rows)


ATTACK_METHODS = {
    "random": random_attack,
    "average": average_attack,
    "bandwagon": bandwagon_attack,
    "segmented": segmented_attack,
}


def mixed_attack(
    train_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    req: AttackRequest,
    weights: Dict[str, float],
) -> pd.DataFrame:
    weights = {k: float(max(0.0, v)) for k, v in weights.items() if k in ATTACK_METHODS}
    if not weights:
        weights = {"random": 1.0}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    n_fake = max(1, int(round(num_users * req.epsilon)))
    methods = list(weights.keys())
    alloc = np.array([weights[m] for m in methods], dtype=np.float64)
    alloc = alloc / alloc.sum()
    counts = np.floor(alloc * n_fake).astype(int)
    while counts.sum() < n_fake:
        counts[int(np.argmax(alloc - counts / max(1, n_fake)))] += 1

    start_uid = next_fake_user_id(train_df, num_users)
    chunks = []
    cursor = start_uid
    for method, cnt in zip(methods, counts):
        if cnt <= 0:
            continue
        fn = ATTACK_METHODS[method]
        chunk = fn(
            train_df=train_df,
            start_user_id=cursor,
            num_fake_users=int(cnt),
            num_items=num_items,
            req=req,
        )
        chunks.append(chunk)
        cursor += int(cnt)

    if not chunks:
        return pd.DataFrame(columns=["user", "item", "rating", "timestamp", "is_malicious"])
    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates(subset=["user", "item"], keep="first").reset_index(drop=True)
    out["user"] = out["user"].astype(int)
    out["item"] = out["item"].astype(int)
    out["rating"] = out["rating"].astype(float)
    out["is_malicious"] = 1
    return out
