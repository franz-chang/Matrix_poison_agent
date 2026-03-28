from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_pos_pairs(train_df: pd.DataFrame, pos_thr: float = 4.0) -> np.ndarray:
    if "rating" not in train_df.columns:
        pairs = train_df[["user", "item"]].astype(int).to_numpy()
        return pairs
    pos = train_df[train_df["rating"] >= pos_thr][["user", "item"]].astype(int).to_numpy()
    if len(pos) == 0:
        pos = train_df[["user", "item"]].astype(int).to_numpy()
    return pos


def _extract_explicit_neg(train_df: pd.DataFrame, neg_thr: float = 2.0) -> Dict[int, set]:
    out: Dict[int, set] = {}
    if "rating" not in train_df.columns:
        return out
    neg_df = train_df[train_df["rating"] <= neg_thr]
    if neg_df.empty:
        return out
    for u, g in neg_df.groupby("user"):
        out[int(u)] = set(g["item"].astype(int).tolist())
    return out


def _sample_neg_item(
    user: int,
    user_pos: Dict[int, set],
    num_items: int,
    rng: np.random.Generator,
    explicit_neg: Optional[Dict[int, set]] = None,
    neg_bias: float = 0.8,
) -> int:
    pos = user_pos.get(int(user), set())
    neg = (explicit_neg or {}).get(int(user), set())
    if neg and rng.random() < neg_bias:
        cand = list(neg - pos)
        if cand:
            return int(cand[int(rng.integers(0, len(cand)))])
    if len(pos) >= num_items:
        return int(rng.integers(0, num_items))
    j = int(rng.integers(0, num_items))
    while j in pos:
        j = int(rng.integers(0, num_items))
    return j


class RecommenderBase:
    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        seed: int,
        device: torch.device,
        positive_threshold: float = 4.0,
        use_explicit_neg: bool = True,
        explicit_neg_bias: float = 0.8,
    ) -> "RecommenderBase":
        raise NotImplementedError

    def score(
        self,
        users: np.ndarray,
        items: np.ndarray,
        user_sequences: Optional[Dict[int, List[int]]] = None,
    ) -> np.ndarray:
        raise NotImplementedError


class BPRMF(RecommenderBase):
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.user_emb: Optional[nn.Embedding] = None
        self.item_emb: Optional[nn.Embedding] = None
        self.device: Optional[torch.device] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        seed: int,
        device: torch.device,
        positive_threshold: float = 4.0,
        use_explicit_neg: bool = True,
        explicit_neg_bias: float = 0.8,
    ) -> "BPRMF":
        _seed_everything(seed)
        self.device = device
        self.user_emb = nn.Embedding(num_users, self.embedding_dim).to(device)
        self.item_emb = nn.Embedding(num_items, self.embedding_dim).to(device)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        optimizer = torch.optim.Adam(
            list(self.user_emb.parameters()) + list(self.item_emb.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        train_pairs = _extract_pos_pairs(train_df, pos_thr=positive_threshold)
        if len(train_pairs) == 0:
            return self
        user_pos: Dict[int, set] = {}
        for u, i in train_pairs:
            user_pos.setdefault(int(u), set()).add(int(i))
        explicit_neg = _extract_explicit_neg(train_df, neg_thr=2.0) if use_explicit_neg else {}

        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            rng.shuffle(train_pairs)
            for start in range(0, len(train_pairs), batch_size):
                batch = train_pairs[start : start + batch_size]
                if len(batch) == 0:
                    continue
                u = batch[:, 0]
                i = batch[:, 1]
                j = np.array(
                    [
                        _sample_neg_item(
                            int(uu),
                            user_pos,
                            num_items,
                            rng,
                            explicit_neg=explicit_neg,
                            neg_bias=explicit_neg_bias,
                        )
                        for uu in u
                    ],
                    dtype=np.int64,
                )
                u_t = torch.tensor(u, dtype=torch.long, device=device)
                i_t = torch.tensor(i, dtype=torch.long, device=device)
                j_t = torch.tensor(j, dtype=torch.long, device=device)

                u_e = self.user_emb(u_t)
                i_e = self.item_emb(i_t)
                j_e = self.item_emb(j_t)
                pos = (u_e * i_e).sum(dim=1)
                neg = (u_e * j_e).sum(dim=1)
                loss = -F.logsigmoid(pos - neg).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def score(
        self,
        users: np.ndarray,
        items: np.ndarray,
        user_sequences: Optional[Dict[int, List[int]]] = None,
    ) -> np.ndarray:
        assert self.user_emb is not None and self.item_emb is not None and self.device is not None
        with torch.no_grad():
            u_t = torch.tensor(users, dtype=torch.long, device=self.device)
            i_t = torch.tensor(items, dtype=torch.long, device=self.device)
            scores = (self.user_emb(u_t) * self.item_emb(i_t)).sum(dim=1).detach().cpu().numpy()
        return scores


class NeuMFNet(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.gmf_user = nn.Embedding(num_users, embedding_dim)
        self.gmf_item = nn.Embedding(num_items, embedding_dim)
        self.mlp_user = nn.Embedding(num_users, embedding_dim)
        self.mlp_item = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(embedding_dim * 2, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user(users) * self.gmf_item(items)
        mlp_in = torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        z = torch.cat([gmf, mlp_out], dim=-1)
        return self.out(z).squeeze(-1)


class NeuMF(RecommenderBase):
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.model: Optional[NeuMFNet] = None
        self.device: Optional[torch.device] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        seed: int,
        device: torch.device,
        positive_threshold: float = 4.0,
        use_explicit_neg: bool = True,
        explicit_neg_bias: float = 0.8,
    ) -> "NeuMF":
        _seed_everything(seed)
        self.device = device
        self.model = NeuMFNet(num_users, num_items, embedding_dim=self.embedding_dim).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        train_pairs = _extract_pos_pairs(train_df, pos_thr=positive_threshold)
        if len(train_pairs) == 0:
            return self
        user_pos: Dict[int, set] = {}
        for u, i in train_pairs:
            user_pos.setdefault(int(u), set()).add(int(i))
        explicit_neg = _extract_explicit_neg(train_df, neg_thr=2.0) if use_explicit_neg else {}
        rng = np.random.default_rng(seed)

        for _ in range(epochs):
            rng.shuffle(train_pairs)
            for start in range(0, len(train_pairs), batch_size):
                batch = train_pairs[start : start + batch_size]
                if len(batch) == 0:
                    continue
                pos_u = batch[:, 0].astype(np.int64)
                pos_i = batch[:, 1].astype(np.int64)
                neg_u = pos_u.copy()
                neg_i = np.array(
                    [
                        _sample_neg_item(
                            int(uu),
                            user_pos,
                            num_items,
                            rng,
                            explicit_neg=explicit_neg,
                            neg_bias=explicit_neg_bias,
                        )
                        for uu in neg_u
                    ],
                    dtype=np.int64,
                )

                users = np.concatenate([pos_u, neg_u])
                items = np.concatenate([pos_i, neg_i])
                labels = np.concatenate([np.ones_like(pos_u), np.zeros_like(neg_u)]).astype(np.float32)

                users_t = torch.tensor(users, dtype=torch.long, device=device)
                items_t = torch.tensor(items, dtype=torch.long, device=device)
                labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
                logits = self.model(users_t, items_t)
                loss = F.binary_cross_entropy_with_logits(logits, labels_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def score(
        self,
        users: np.ndarray,
        items: np.ndarray,
        user_sequences: Optional[Dict[int, List[int]]] = None,
    ) -> np.ndarray:
        assert self.model is not None and self.device is not None
        with torch.no_grad():
            u_t = torch.tensor(users, dtype=torch.long, device=self.device)
            i_t = torch.tensor(items, dtype=torch.long, device=self.device)
            out = self.model(u_t, i_t).detach().cpu().numpy()
        return out


class LightGCNNet(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def propagate(self, norm_adj: torch.Tensor) -> torch.Tensor:
        embs = [self.embedding.weight]
        x = self.embedding.weight
        for _ in range(self.n_layers):
            x = torch.sparse.mm(norm_adj, x)
            embs.append(x)
        return torch.stack(embs, dim=0).mean(dim=0)


class LightGCN(RecommenderBase):
    def __init__(self, embedding_dim: int = 64, n_layers: int = 2):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.model: Optional[LightGCNNet] = None
        self.norm_adj: Optional[torch.Tensor] = None
        self.num_users: int = 0
        self.num_items: int = 0
        self.device: Optional[torch.device] = None

    @staticmethod
    def _build_norm_adj(num_users: int, num_items: int, pairs: np.ndarray, device: torch.device) -> torch.Tensor:
        n_nodes = num_users + num_items
        users = pairs[:, 0].astype(np.int64)
        items = pairs[:, 1].astype(np.int64) + num_users
        src = np.concatenate([users, items])
        dst = np.concatenate([items, users])
        vals = np.ones_like(src, dtype=np.float32)
        idx = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=device)
        vals_t = torch.tensor(vals, dtype=torch.float32, device=device)
        adj = torch.sparse_coo_tensor(idx, vals_t, size=(n_nodes, n_nodes)).coalesce()

        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        row, col = adj.indices()
        norm_vals = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(adj.indices(), norm_vals, adj.size(), device=device).coalesce()

    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        seed: int,
        device: torch.device,
        positive_threshold: float = 4.0,
        use_explicit_neg: bool = True,
        explicit_neg_bias: float = 0.8,
    ) -> "LightGCN":
        _seed_everything(seed)
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        self.model = LightGCNNet(num_users, num_items, embedding_dim=self.embedding_dim, n_layers=self.n_layers).to(device)

        pairs = _extract_pos_pairs(train_df, pos_thr=positive_threshold)
        if len(pairs) == 0:
            return self
        self.norm_adj = self._build_norm_adj(num_users, num_items, pairs, device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        user_pos: Dict[int, set] = {}
        for u, i in pairs:
            user_pos.setdefault(int(u), set()).add(int(i))
        explicit_neg = _extract_explicit_neg(train_df, neg_thr=2.0) if use_explicit_neg else {}
        rng = np.random.default_rng(seed)

        for _ in range(epochs):
            rng.shuffle(pairs)
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                if len(batch) == 0:
                    continue
                users = batch[:, 0]
                pos_items = batch[:, 1]
                neg_items = np.array(
                    [
                        _sample_neg_item(
                            int(u),
                            user_pos,
                            num_items,
                            rng,
                            explicit_neg=explicit_neg,
                            neg_bias=explicit_neg_bias,
                        )
                        for u in users
                    ],
                    dtype=np.int64,
                )

                all_emb = self.model.propagate(self.norm_adj)
                user_emb = all_emb[torch.tensor(users, dtype=torch.long, device=device)]
                pos_emb = all_emb[torch.tensor(pos_items + num_users, dtype=torch.long, device=device)]
                neg_emb = all_emb[torch.tensor(neg_items + num_users, dtype=torch.long, device=device)]
                pos_scores = (user_emb * pos_emb).sum(dim=1)
                neg_scores = (user_emb * neg_emb).sum(dim=1)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def score(
        self,
        users: np.ndarray,
        items: np.ndarray,
        user_sequences: Optional[Dict[int, List[int]]] = None,
    ) -> np.ndarray:
        assert self.model is not None and self.norm_adj is not None and self.device is not None
        with torch.no_grad():
            all_emb = self.model.propagate(self.norm_adj)
            u_t = torch.tensor(users, dtype=torch.long, device=self.device)
            i_t = torch.tensor(items + self.num_users, dtype=torch.long, device=self.device)
            scores = (all_emb[u_t] * all_emb[i_t]).sum(dim=1).detach().cpu().numpy()
        return scores


class SASRecNet(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 64,
        max_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.pad_idx = num_items
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=self.pad_idx)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def encode(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = seq.shape
        pos = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(bsz, -1)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.dropout(x)
        pad_mask = seq.eq(self.pad_idx)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return h, pad_mask

    def final_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        h, pad_mask = self.encode(seq)
        valid_len = (~pad_mask).sum(dim=1).clamp(min=1) - 1
        idx = torch.arange(seq.size(0), device=seq.device)
        return h[idx, valid_len]


class SASRec(RecommenderBase):
    def __init__(self, hidden_size: int = 64, max_len: int = 50, num_layers: int = 2, num_heads: int = 2):
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model: Optional[SASRecNet] = None
        self.num_items: int = 0
        self.device: Optional[torch.device] = None
        self.train_sequences: Dict[int, List[int]] = {}

    def _build_one_step_samples(self, train_df: pd.DataFrame) -> List[Tuple[int, List[int], int]]:
        samples: List[Tuple[int, List[int], int]] = []
        for user, g in train_df.groupby("user"):
            seq = g.sort_values("timestamp")["item"].astype(int).tolist()
            if len(seq) < 2:
                continue
            # Use a handful of recent next-item transitions to avoid over-skewing
            # training toward very long histories.
            recent_steps = min(5, len(seq) - 1)
            start = len(seq) - recent_steps
            for pos in range(start, len(seq)):
                hist = seq[max(0, pos - self.max_len):pos]
                target = seq[pos]
                samples.append((int(user), hist, int(target)))
        return samples

    def _pad(self, seq: Sequence[int]) -> List[int]:
        if len(seq) >= self.max_len:
            return list(seq[-self.max_len :])
        return [self.num_items] * (self.max_len - len(seq)) + list(seq)

    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        seed: int,
        device: torch.device,
        positive_threshold: float = 4.0,
        use_explicit_neg: bool = True,
        explicit_neg_bias: float = 0.8,
    ) -> "SASRec":
        _seed_everything(seed)
        self.device = device
        self.num_items = num_items
        self.model = SASRecNet(
            num_items=num_items,
            hidden_size=self.hidden_size,
            max_len=self.max_len,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        ).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        train_pos_df = (
            train_df[train_df["rating"] >= positive_threshold].copy()
            if "rating" in train_df.columns
            else train_df.copy()
        )
        if train_pos_df.empty:
            train_pos_df = train_df.copy()
        self.train_sequences = {
            int(u): g.sort_values("timestamp")["item"].astype(int).tolist()[-self.max_len :]
            for u, g in train_pos_df.groupby("user")
        }
        samples = self._build_one_step_samples(train_pos_df)
        if not samples:
            return self

        user_pos: Dict[int, set] = {}
        for u, g in train_pos_df.groupby("user"):
            user_pos[int(u)] = set(g["item"].astype(int).tolist())
        explicit_neg = _extract_explicit_neg(train_df, neg_thr=2.0) if use_explicit_neg else {}
        rng = np.random.default_rng(seed)

        for _ in range(epochs):
            rng.shuffle(samples)
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                if not batch:
                    continue
                seqs = np.array([self._pad(hist) for _, hist, _ in batch], dtype=np.int64)
                pos_items = np.array([target for _, _, target in batch], dtype=np.int64)
                users = np.array([u for u, _, _ in batch], dtype=np.int64)
                neg_items = np.array(
                    [
                        _sample_neg_item(
                            int(u),
                            user_pos,
                            num_items,
                            rng,
                            explicit_neg=explicit_neg,
                            neg_bias=explicit_neg_bias,
                        )
                        for u in users
                    ],
                    dtype=np.int64,
                )

                seq_t = torch.tensor(seqs, dtype=torch.long, device=device)
                pos_t = torch.tensor(pos_items, dtype=torch.long, device=device)
                neg_t = torch.tensor(neg_items, dtype=torch.long, device=device)
                h = self.model.final_hidden(seq_t)
                pos_e = self.model.item_emb(pos_t)
                neg_e = self.model.item_emb(neg_t)
                pos_logits = (h * pos_e).sum(dim=1)
                neg_logits = (h * neg_e).sum(dim=1)
                loss = (
                    F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                    + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def score(
        self,
        users: np.ndarray,
        items: np.ndarray,
        user_sequences: Optional[Dict[int, List[int]]] = None,
    ) -> np.ndarray:
        assert self.model is not None and self.device is not None
        seq_store = user_sequences if user_sequences is not None else self.train_sequences
        seqs = []
        for u in users:
            seq = seq_store.get(int(u), [])
            seqs.append(self._pad(seq))
        with torch.no_grad():
            seq_t = torch.tensor(np.array(seqs, dtype=np.int64), dtype=torch.long, device=self.device)
            h = self.model.final_hidden(seq_t)
            i_t = torch.tensor(items, dtype=torch.long, device=self.device)
            item_emb = self.model.item_emb(i_t)
            scores = (h * item_emb).sum(dim=1).detach().cpu().numpy()
        return scores


@dataclass
class TrainConfig:
    model_name: str = "bpr"
    embedding_dim: int = 64
    lightgcn_layers: int = 2
    sasrec_max_len: int = 50
    sasrec_heads: int = 2
    sasrec_layers: int = 2
    epochs: int = 20
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-6
    seed: int = 42
    device: str = "cpu"
    positive_threshold: float = 4.0
    use_explicit_neg: bool = True
    explicit_neg_bias: float = 0.8


def build_model(cfg: TrainConfig) -> RecommenderBase:
    name = cfg.model_name.lower()
    if name == "bpr":
        return BPRMF(embedding_dim=cfg.embedding_dim)
    if name == "neumf":
        return NeuMF(embedding_dim=cfg.embedding_dim)
    if name == "lightgcn":
        return LightGCN(embedding_dim=cfg.embedding_dim, n_layers=cfg.lightgcn_layers)
    if name == "sasrec":
        return SASRec(
            hidden_size=cfg.embedding_dim,
            max_len=cfg.sasrec_max_len,
            num_layers=cfg.sasrec_layers,
            num_heads=cfg.sasrec_heads,
        )
    raise ValueError(f"Unsupported model: {cfg.model_name}")


def train_model(
    train_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    cfg: TrainConfig,
) -> RecommenderBase:
    model = build_model(cfg)
    device = torch.device(cfg.device)
    model.fit(
        train_df=train_df,
        num_users=num_users,
        num_items=num_items,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        device=device,
        positive_threshold=cfg.positive_threshold,
        use_explicit_neg=cfg.use_explicit_neg,
        explicit_neg_bias=cfg.explicit_neg_bias,
    )
    return model
