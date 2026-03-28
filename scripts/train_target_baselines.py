#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_poison.data import prepare_data
from matrix_poison.models import TrainConfig, train_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and save baseline target-model checkpoints.")
    p.add_argument("--interactions", default="data/ml1m/interactions.csv")
    p.add_argument("--items", default="data/ml1m/items.csv")
    p.add_argument("--output-dir", default="checkpoints/target_models/ml1m")
    p.add_argument("--models", default="bpr,neumf,lightgcn,sasrec")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--sasrec-batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lightgcn-layers", type=int, default=2)
    p.add_argument("--sasrec-max-len", type=int, default=50)
    p.add_argument("--sasrec-heads", type=int, default=2)
    p.add_argument("--sasrec-layers", type=int, default=2)
    p.add_argument("--min-positive-rating", type=float, default=1.0)
    p.add_argument("--positive-threshold", type=float, default=4.0)
    p.add_argument("--use-explicit-neg", action="store_true")
    p.add_argument("--explicit-neg-bias", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto/cpu/cuda")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_state_dict(model_name: str, model_obj) -> Dict[str, torch.Tensor]:
    name = model_name.lower()
    if name == "bpr":
        if getattr(model_obj, "user_emb", None) is None or getattr(model_obj, "item_emb", None) is None:
            raise RuntimeError("BPR model embeddings are not initialized.")
        return {
            "user_emb.weight": model_obj.user_emb.weight.detach().cpu(),
            "item_emb.weight": model_obj.item_emb.weight.detach().cpu(),
        }

    module = getattr(model_obj, "model", None)
    if module is None:
        raise RuntimeError(f"model `{model_name}` has no torch module to export.")
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data(
        interactions_path=args.interactions,
        items_path=args.items,
        min_positive_rating=args.min_positive_rating,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    models: List[str] = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    manifest = {
        "interactions": str(Path(args.interactions).resolve()),
        "items": str(Path(args.items).resolve()),
        "num_users": int(data.num_users),
        "num_items": int(data.num_items),
        "models": {},
    }

    for idx, model_name in enumerate(models):
        batch_size = args.sasrec_batch_size if model_name == "sasrec" else args.batch_size
        cfg = TrainConfig(
            model_name=model_name,
            embedding_dim=args.embedding_dim,
            lightgcn_layers=args.lightgcn_layers,
            sasrec_max_len=args.sasrec_max_len,
            sasrec_heads=args.sasrec_heads,
            sasrec_layers=args.sasrec_layers,
            epochs=args.epochs,
            batch_size=batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + idx,
            device=device,
            positive_threshold=args.positive_threshold,
            use_explicit_neg=args.use_explicit_neg,
            explicit_neg_bias=args.explicit_neg_bias,
        )

        print(f"[baseline] training {model_name} (epochs={cfg.epochs}, batch={cfg.batch_size}, device={cfg.device})")
        model_obj = train_model(
            train_df=data.train,
            num_users=data.num_users,
            num_items=data.num_items,
            cfg=cfg,
        )

        ckpt = {
            "model_name": model_name,
            "state_dict": extract_state_dict(model_name, model_obj),
            "num_users": int(data.num_users),
            "num_items": int(data.num_items),
            "train_config": asdict(cfg),
            "dataset": {
                "interactions": str(Path(args.interactions).resolve()),
                "items": str(Path(args.items).resolve()),
            },
        }
        ckpt_path = out_dir / f"{model_name}_baseline.pt"
        torch.save(ckpt, ckpt_path)
        print(f"[baseline] saved -> {ckpt_path}")

        manifest["models"][model_name] = {
            "checkpoint": str(ckpt_path),
            "train_config": asdict(cfg),
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[baseline] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
