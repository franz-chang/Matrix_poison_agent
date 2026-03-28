from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert MovieLens-1M raw files into Matrix-Poison CSV format.")
    p.add_argument("--ratings-dat", required=True, help="Path to ratings.dat")
    p.add_argument("--movies-dat", required=True, help="Path to movies.dat")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(
        args.ratings_dat,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        args.movies_dat,
        sep="::",
        engine="python",
        names=["item_id", "title", "genre"],
        encoding="latin-1",
    )
    items = movies[["item_id", "genre"]].copy()
    # Use the first genre as a category key.
    items["category"] = items["genre"].astype(str).str.split("|").str[0]
    items = items[["item_id", "category"]]

    ratings.to_csv(out / "interactions.csv", index=False)
    items.to_csv(out / "items.csv", index=False)
    print(f"Saved processed files to: {out.resolve()}")


if __name__ == "__main__":
    main()
