#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MARKETS = {
    "us": "US",
    "jp": "JP",
    "uk": "UK",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare XMRec Amazon Musical US/JP/UK into interactions/items CSVs.")
    p.add_argument("--raw-root", default="data/xmrec_musical/raw")
    p.add_argument("--output-root", default="data/xmrec_musical/markets")
    p.add_argument("--category-name", default="Musical Instruments")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def convert_one(raw_file: Path, out_dir: Path, category_name: str) -> dict:
    df = pd.read_csv(
        raw_file,
        sep=r"\s+",
        header=None,
        names=["user_id", "item_id", "rating", "date"],
        compression="gzip",
        dtype={"user_id": "string", "item_id": "string", "rating": "float64", "date": "string"},
    )
    ts = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"{raw_file} has {bad} unparsable dates")

    # Convert date to Unix seconds robustly regardless of pandas datetime internal unit.
    df["timestamp"] = ts.astype("datetime64[s]").astype("int64")
    inter = df[["user_id", "item_id", "rating", "timestamp"]].copy()
    inter["user_id"] = inter["user_id"].astype(str).str.strip()
    inter["item_id"] = inter["item_id"].astype(str).str.strip()
    inter = inter.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)

    items = pd.DataFrame({"item_id": sorted(inter["item_id"].unique().tolist())})
    items["category"] = category_name

    out_dir.mkdir(parents=True, exist_ok=True)
    inter.to_csv(out_dir / "interactions.csv", index=False)
    items.to_csv(out_dir / "items.csv", index=False)

    return {
        "rows": int(len(inter)),
        "users": int(inter["user_id"].nunique()),
        "items": int(inter["item_id"].nunique()),
        "min_rating": float(inter["rating"].min()),
        "max_rating": float(inter["rating"].max()),
        "min_date": str(pd.to_datetime(inter["timestamp"], unit="s").min().date()),
        "max_date": str(pd.to_datetime(inter["timestamp"], unit="s").max().date()),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    raw_root = (root / args.raw_root).resolve()
    out_root = (root / args.output_root).resolve()

    out_root.mkdir(parents=True, exist_ok=True)

    for code, market in MARKETS.items():
        src = raw_root / f"ratings_{code}_Musical_Instruments.txt.gz"
        dst = out_root / market
        inter = dst / "interactions.csv"
        items = dst / "items.csv"

        if (not args.force) and inter.exists() and items.exists():
            print(f"[skip] {market}: found {inter} and {items}")
            continue

        if not src.exists():
            raise FileNotFoundError(f"missing raw file: {src}")

        stats = convert_one(src, dst, args.category_name)
        print(
            f"[ok] {market}: rows={stats['rows']} users={stats['users']} items={stats['items']} "
            f"rating=[{stats['min_rating']:.1f},{stats['max_rating']:.1f}] "
            f"date=[{stats['min_date']}..{stats['max_date']}]"
        )


if __name__ == "__main__":
    main()
