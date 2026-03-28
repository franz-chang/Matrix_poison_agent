from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a small synthetic recommender dataset.")
    p.add_argument("--output-dir", default="examples/synth")
    p.add_argument("--num-users", type=int, default=500)
    p.add_argument("--num-items", type=int, default=800)
    p.add_argument("--num-categories", type=int, default=8)
    p.add_argument("--avg-interactions", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    categories = [f"C{i}" for i in range(args.num_categories)]
    item_ids = [f"item_{i}" for i in range(args.num_items)]
    item_category = rng.choice(categories, size=args.num_items, replace=True)
    items_df = pd.DataFrame({"item_id": item_ids, "category": item_category})
    items_df.to_csv(out / "items.csv", index=False)

    rows = []
    ts = 1_700_000_000
    for u in range(args.num_users):
        n = int(max(5, rng.normal(args.avg_interactions, args.avg_interactions * 0.2)))
        user_items = rng.choice(args.num_items, size=min(n, args.num_items), replace=False)
        for i in user_items:
            cat = item_category[i]
            # Slight category preference to keep structure.
            bias = 0.5 if cat == categories[u % args.num_categories] else 0.0
            rating = float(np.clip(rng.normal(3.8 + bias, 0.8), 1.0, 5.0))
            rows.append(
                {
                    "user_id": f"user_{u}",
                    "item_id": item_ids[i],
                    "rating": rating,
                    "timestamp": ts,
                }
            )
            ts += int(rng.integers(1, 10))
    inter_df = pd.DataFrame(rows)
    inter_df.to_csv(out / "interactions.csv", index=False)
    print(f"Saved synthetic dataset to: {out.resolve()}")


if __name__ == "__main__":
    main()
