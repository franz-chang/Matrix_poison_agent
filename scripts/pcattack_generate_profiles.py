#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate PC-Attack-style malicious profiles (lightweight adapter).")
    p.add_argument("--pcattack-root", required=True)
    p.add_argument("--dataset-alias", required=True)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--target-item", type=int, required=True)
    p.add_argument("--num-fake-user", type=int, required=True)
    p.add_argument("--num-filler-item", type=int, default=50)
    p.add_argument("--num-filler-pop", type=int, default=30)
    p.add_argument("--num-epochs", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--intent", choices=["nuke", "push"], default="nuke")
    p.add_argument("--output-csv", required=True)
    return p.parse_args()


def _weighted_sample_without_replacement(
    rng: np.random.Generator,
    pool: np.ndarray,
    probs: np.ndarray,
    k: int,
) -> np.ndarray:
    if k <= 0 or len(pool) == 0:
        return np.array([], dtype=np.int64)
    k = min(int(k), int(len(pool)))
    p = probs.astype(np.float64)
    p = np.clip(p, 0.0, None)
    if p.sum() <= 0:
        p = np.ones_like(p, dtype=np.float64) / len(p)
    else:
        p = p / p.sum()
    return rng.choice(pool, size=k, replace=False, p=p)


def _build_pcattack_lite(
    train_df: pd.DataFrame,
    target_item: int,
    num_fake_user: int,
    num_filler_item: int,
    num_filler_pop: int,
    intent: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(1234)

    train = train_df[["user", "item", "rating"]].copy()
    train["user"] = train["user"].astype(int)
    train["item"] = train["item"].astype(int)
    train["rating"] = train["rating"].astype(float)

    n_users = int(train["user"].max()) + 1 if not train.empty else 1

    item_pop = train["item"].value_counts().sort_index()
    pop_items = item_pop.index.to_numpy(dtype=np.int64)
    pop_counts = item_pop.to_numpy(dtype=np.float64)

    if len(pop_items) == 0:
        return pd.DataFrame(columns=["user", "item", "rating", "timestamp", "is_malicious"])

    user_items = {}
    for u, g in train.groupby("user"):
        user_items[int(u)] = set(g["item"].astype(int).tolist())

    target_users = [u for u, items in user_items.items() if int(target_item) in items]
    if not target_users:
        target_users = list(user_items.keys())

    pop_pool = pop_items[pop_items != int(target_item)]
    pop_w = pop_counts[pop_items != int(target_item)]

    rows = []
    for fake_idx in range(int(num_fake_user)):
        fake_user = n_users + fake_idx
        chosen = set()
        if intent == "push":
            chosen.add(int(target_item))

        if len(pop_pool) > 0 and num_filler_pop > 0:
            sampled_pop = _weighted_sample_without_replacement(
                rng,
                pop_pool,
                pop_w,
                int(num_filler_pop),
            )
            chosen.update(int(x) for x in sampled_pop.tolist())

        anchor = int(target_users[int(rng.integers(0, len(target_users)))])
        anchor_items = [i for i in user_items.get(anchor, set()) if int(i) != int(target_item)]
        if anchor_items:
            anchor_items = sorted(anchor_items, key=lambda i: item_pop.get(int(i), 0), reverse=True)
            anchor_items = anchor_items[: int(max(0, num_filler_item))]
            chosen.update(int(i) for i in anchor_items)

        all_items = list(chosen)
        for item in all_items:
            if intent == "push" and int(item) == int(target_item):
                rating = float(np.clip(rng.normal(4.95, 0.08), 4.6, 5.0))
            elif intent == "push":
                rating = float(np.clip(rng.normal(4.2, 0.2), 3.8, 5.0))
            else:
                # Nuke mode for implicit-feedback victims: avoid interacting with target item.
                rating = float(np.clip(rng.normal(4.6, 0.2), 4.0, 5.0))
            rows.append(
                {
                    "user": int(fake_user),
                    "item": int(item),
                    "rating": rating,
                    "timestamp": np.nan,
                    "is_malicious": 1,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["user", "item"], keep="first").reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    need = {"user", "item", "rating"}
    if not need.issubset(set(train_df.columns)):
        raise ValueError(f"train csv missing columns: {need - set(train_df.columns)}")

    out_df = _build_pcattack_lite(
        train_df=train_df,
        target_item=int(args.target_item),
        num_fake_user=int(args.num_fake_user),
        num_filler_item=int(args.num_filler_item),
        num_filler_pop=int(args.num_filler_pop),
        intent=str(args.intent).lower(),
    )

    out_path = Path(args.output_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(
        json.dumps(
            {
                "mode": "pcattack_lite",
                "dataset": args.dataset_alias,
                "target_item": int(args.target_item),
                "intent": str(args.intent).lower(),
                "num_fake_users": int(args.num_fake_user),
                "rows": int(len(out_df)),
                "output_csv": str(out_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
