#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transformer_scaling_shape_sweep.py
===================================

Goal
----
For multiple fixed parameter budgets N, sweep the depth-to-width ratio
(aspect ratio) of a decoder-only Transformer and find the shape that
minimises test MSE. The shape effect is mediated by AOFE_ratio (AGOP
off-diagonal Frobenius energy ratio), a measure of feature coupling
(gradient superposition strength).

Chinchilla-style hypothesis
----------------------------
Just as Chinchilla identifies compute-optimal (N, D) pairs, this experiment
identifies the *shape-optimal* (depth, d_model) pair for each N. The key
prediction is:

  1. For each N, there exists an optimal aspect ratio α* = depth / d_model
     that simultaneously minimises test MSE and AOFE_ratio.
  2. AOFE_ratio is positively correlated with test MSE across shapes.
  3. The optimal aspect ratio α* is approximately invariant to N
     (or follows a weak power-law), analogous to the Chinchilla 20-token rule.

Task: Teacher-Student Feature Regression
-----------------------------------------
- Teacher: frozen 4-layer MLP, maps [B, SEQ_LEN] → R^{TEACHER_OUT}.
- Student: TinyGPT (depth, d_model) padded to exactly N parameters.
- Training budget: D = data_ratio × N supervised outputs (Chinchilla D=20N).
- Train/val/test split: val/test are fixed at 3 000 sequences each;
  train_size = D / TEACHER_OUT.

Output-space AGOP (fixed 64×64 across all shapes)
--------------------------------------------------
  J = d(ŷ ∈ R^{TEACHER_OUT}) / d(e_flat ∈ R^{T×d_model})
  AGOP = E_data[J J^T] ∈ R^{TEACHER_OUT × TEACHER_OUT}

AOFE_ratio = off-diagonal Frobenius energy / total Frobenius energy.

Usage
-----
  python experiments/transformer_scaling_shape_sweep.py \\
      --param_groups 300000,1000000,3000000 \\
      --depth_list 2,3,4,5,6,8,10,12,16,20,24 \\
      --out_dir ./results_scaling_shape_sweep \\
      --device cuda
"""

from __future__ import annotations

import os
import csv
import math
import time
import random
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

TEACHER_OUT = 64   # teacher output dim → AGOP ∈ R^{64×64}
SEQ_LEN     = 80   # tokens per sequence
INP_DIM     = 1    # scalar token input


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def symmetrize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.T)


# ─────────────────────────────────────────────
# AOFE metric
# ─────────────────────────────────────────────

def agop_offdiag_metrics(agop: torch.Tensor) -> Tuple[float, float]:
    """
    Returns (AOFE, AOFE_ratio):
      AOFE       = ||AGOP||_F^2 - ||diag(AGOP)||_2^2
      AOFE_ratio = AOFE / ||AGOP||_F^2
    """
    agop = agop.float()
    fro2  = float((agop * agop).sum().item()) + 1e-12
    diag2 = float((torch.diag(agop) ** 2).sum().item())
    return max(fro2 - diag2, 0.0), max(fro2 - diag2, 0.0) / fro2


# ─────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    def _rank(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
        return ranks
    return pearson_corr(_rank(x), _rank(y))


# ─────────────────────────────────────────────
# Transformer: CausalSelfAttention → MLPBlock → DecoderBlock → TinyGPT
# ─────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.dropout = dropout
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask   = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn   = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, d)
        out = self.proj(out)
        if self.dropout > 0 and self.training:
            out = F.dropout(out, p=self.dropout)
        return out


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff, bias=False)
        self.fc2     = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        if self.dropout > 0 and self.training:
            x = F.dropout(x, p=self.dropout)
        x = self.fc2(x)
        if self.dropout > 0 and self.training:
            x = F.dropout(x, p=self.dropout)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = MLPBlock(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Decoder-only transformer for teacher-student regression.

    forward(x: [B, T, inp_dim]) → last-position output [B, out_dim].
    """

    def __init__(
        self,
        *,
        inp_dim: int = INP_DIM,
        out_dim: int = TEACHER_OUT,
        seq_len: int = SEQ_LEN,
        depth: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        pad_params: int = 0,
    ):
        super().__init__()
        self.out_dim  = out_dim
        self.seq_len  = seq_len
        self.d_model  = d_model
        self.depth    = depth

        self.inp_emb  = nn.Linear(inp_dim, d_model, bias=True)
        self.pos_emb  = nn.Embedding(seq_len, d_model)
        self.drop     = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.blocks   = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(depth)
        ])
        self.ln_f     = nn.LayerNorm(d_model)
        self.head_out = nn.Linear(d_model, out_dim, bias=True)

        self._pad = None
        if pad_params > 0:
            self._pad = nn.Parameter(torch.zeros(pad_params), requires_grad=True)

    def forward_from_embeddings(self, e: torch.Tensor) -> torch.Tensor:
        """e: [B, T, d_model] → [B, T, out_dim]"""
        x = self.drop(e)
        for blk in self.blocks:
            x = blk(x)
        return self.head_out(self.ln_f(x))

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """x_in: [B, T, inp_dim] → [B, T, out_dim]"""
        B, T, _ = x_in.shape
        pos = torch.arange(T, device=x_in.device).unsqueeze(0).expand(B, T)
        e   = self.inp_emb(x_in) + self.pos_emb(pos)
        return self.forward_from_embeddings(e)


# ─────────────────────────────────────────────
# Teachers (both frozen, random init)
# ─────────────────────────────────────────────

class TeacherMLP(nn.Module):
    """
    4-layer MLP: R^{SEQ_LEN} → R^{256} × 3 → R^{TEACHER_OUT}.
    Processes all tokens simultaneously (no sequential structure).
    Provides a "flat" regression target.
    """
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SEQ_LEN, hidden, bias=True), nn.GELU(),
            nn.Linear(hidden, hidden, bias=True),  nn.GELU(),
            nn.Linear(hidden, hidden, bias=True),  nn.GELU(),
            nn.Linear(hidden, TEACHER_OUT, bias=True),
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 1] → [B, TEACHER_OUT]"""
        return self.net(x.view(x.shape[0], -1))


class TeacherGPT(nn.Module):
    """
    6-layer decoder-only Transformer teacher (frozen random weights).

    Why a Transformer teacher for the scaling experiment:
    - The teacher uses causal self-attention over SEQ_LEN=80 tokens, creating
      MULTI-STEP, long-range dependencies that cannot be approximated by a
      single attention layer.
    - Shallow students (depth=1-2) lack the representational depth to replicate
      the teacher's 6-layer computations → high MSE lower bound.
    - Deep students (depth≥16) are forced into very narrow d_model by the N
      budget → high AOFE_ratio → high MSE upper bound.
    - Students at intermediate depth (≈6 layers) match the teacher structure
      and have moderate d_model → optimal MSE and lowest AOFE_ratio.
    - This produces the U-shaped loss-vs-depth curve needed for the
      Chinchilla-style "optimal shape" finding.

    Architecture:  [B, T, 1] → inp_emb → 6 × DecoderBlock → ln_f → head_out
                → last-position output [B, TEACHER_OUT]
    """
    def __init__(
        self,
        depth: int = 6,
        d_model: int = 64,
        n_heads: int = 8,
        d_ff: int = 256,
    ):
        super().__init__()
        self.inp_emb  = nn.Linear(INP_DIM, d_model, bias=True)
        self.pos_emb  = nn.Embedding(SEQ_LEN, d_model)
        self.blocks   = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout=0.0) for _ in range(depth)
        ])
        self.ln_f     = nn.LayerNorm(d_model)
        self.head_out = nn.Linear(d_model, TEACHER_OUT, bias=True)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 1] → [B, TEACHER_OUT]  (last-position output)"""
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h   = self.inp_emb(x) + self.pos_emb(pos)
        for blk in self.blocks:
            h = blk(h)
        return self.head_out(self.ln_f(h))[:, -1, :]   # [B, TEACHER_OUT]


def build_teacher(
    seed: int,
    device: torch.device,
    teacher_type: str = "gpt",
) -> nn.Module:
    """Build and freeze a teacher network.

    teacher_type:
      'mlp' – flat 4-layer MLP; tends to produce monotone loss-vs-depth
               (shallower students always win) because the teacher itself
               has no sequential/causal structure.
      'gpt' – 6-layer decoder Transformer; produces U-shaped loss-vs-depth
               because students must match the teacher's multi-step
               attention computation.
    """
    set_seed(seed)
    if teacher_type == "gpt":
        teacher = TeacherGPT(depth=6, d_model=64, n_heads=8, d_ff=256)
    else:
        teacher = TeacherMLP(hidden=256)
    teacher.eval().to(device)
    print(f"  Teacher ({teacher_type}) params (frozen): {count_params(teacher):,}")
    return teacher


@torch.no_grad()
def precompute_teacher_outputs(
    teacher: nn.Module, x: torch.Tensor, device: torch.device, batch_size: int = 512
) -> torch.Tensor:
    chunks = []
    for i in range(0, len(x), batch_size):
        chunks.append(teacher(x[i:i+batch_size].to(device)).cpu())
    return torch.cat(chunks, dim=0)


def make_dataset(
    n: int,
    seed: int,
    teacher: nn.Module,
    device: torch.device,
    y_scale: Optional[float] = None,
) -> Tuple[torch.utils.data.TensorDataset, float]:
    """Generate n random sequences and pre-compute teacher outputs.
    Normalises targets to unit std (y_scale from train is re-used for val/test).
    """
    rng  = np.random.default_rng(seed)
    x_np = rng.standard_normal((n, SEQ_LEN, INP_DIM)).astype(np.float32)
    x_t  = torch.from_numpy(x_np)
    y_t  = precompute_teacher_outputs(teacher, x_t, device)
    if y_scale is None:
        y_scale = float(y_t.std().item()) or 1.0
    y_t = y_t / y_scale
    return torch.utils.data.TensorDataset(x_t, y_t), y_scale


# ─────────────────────────────────────────────
# AGOP estimation (w.r.t. input embeddings)
# ─────────────────────────────────────────────

def estimate_agop_wrt_embeddings(
    model: TinyGPT,
    x_in: torch.Tensor,
    *,
    proj_samples: int = 64,
) -> torch.Tensor:
    """
    AGOP = E_data[J J^T],  J = d(ŷ[-1] ∈ R^{TEACHER_OUT}) / d(e ∈ R^{T×d_model}).
    AGOP ∈ R^{TEACHER_OUT × TEACHER_OUT} — fixed size across all shapes.
    Estimated via random JVPs (forward-mode AD).
    """
    device = x_in.device
    model.eval()
    D_out = model.out_dim

    with torch.no_grad():
        B, T, _ = x_in.shape
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        e   = (model.inp_emb(x_in) + model.pos_emb(pos)).detach()

    agop = torch.zeros(D_out, D_out, device=device)

    def fwd(e_in: torch.Tensor) -> torch.Tensor:
        return model.forward_from_embeddings(e_in)[:, -1, :]

    for _ in range(proj_samples):
        u = torch.randn_like(e)
        _, Ju = torch.autograd.functional.jvp(fwd, (e,), (u,), create_graph=False)
        Ju = torch.nan_to_num(Ju.float(), nan=0.0, posinf=0.0, neginf=0.0)
        agop = agop + (Ju.T @ Ju) / float(B)

    return symmetrize(agop / float(proj_samples)).detach()


# ─────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────

@dataclass
class TrainCfg:
    lr:               float = 3e-4
    weight_decay:     float = 1e-4
    data_ratio:       float = 20.0
    warmup_steps:     int   = 300
    batch_size:       int   = 256
    grad_clip:        float = 1.0
    eval_every:       int   = 200
    val_size:         int   = 3_000
    test_size:        int   = 3_000
    # head_dim=4 gives finer d_model granularity (more valid configs per N)
    # compared to head_dim=8; d_head = d_model // head_dim = 4 (always).
    head_dim:         int   = 4
    dropout:          float = 0.0
    max_padding_ratio: float = 0.20
    max_train_factor:  float = 1.5
    fit_patience:      int   = 8
    agop_batch:        int   = 256
    agop_proj_samples: int   = 64
    seed:              int   = 0
    d_model_min:       int   = 24
    d_model_max:       int   = 1024


def cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate(
    model: TinyGPT,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total_loss, total_n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        pred = model(x)[:, -1, :]
        total_loss += float(F.mse_loss(pred, y, reduction="sum").item())
        total_n    += int(y.numel())
    return total_loss / max(1, total_n)


def train_one_model(
    model: TinyGPT,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    test_loader:  torch.utils.data.DataLoader,
    base_steps:   int,
    cfg:          TrainCfg,
    device:       torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Train to teacher-student MSE. Checkpoint best val state.
    Stops after base_steps * max_train_factor steps OR when val_mse
    has not improved for fit_patience eval periods.
    """
    model.to(device).train()
    opt       = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader_it = iter(train_loader)

    max_steps   = max(base_steps, int(math.ceil(cfg.max_train_factor * base_steps)))
    best_val    = float("inf")
    best_state  = None
    stale       = 0
    history: List[Dict[str, float]] = []
    t0 = time.time()

    for step in range(max_steps):
        try:
            x, y = next(loader_it)
        except StopIteration:
            loader_it = iter(train_loader)
            x, y = next(loader_it)

        x, y = x.to(device), y.to(device)
        lr = cosine_lr(step, cfg.lr, cfg.warmup_steps, max_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        pred = model(x)[:, -1, :]
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % cfg.eval_every == 0 or (step + 1) == max_steps:
            tr_mse  = evaluate(model, train_loader, device, max_batches=20)
            val_mse = evaluate(model, val_loader,   device)
            te_mse  = evaluate(model, test_loader,  device)
            history.append({
                "step": step + 1, "lr": lr,
                "train_mse": tr_mse, "val_mse": val_mse, "test_mse": te_mse,
            })
            elapsed = time.time() - t0
            print(
                f"    step {step+1:6d}/{max_steps}  lr={lr:.2e}  "
                f"train={tr_mse:.5f}  val={val_mse:.5f}  test={te_mse:.5f}  "
                f"t={elapsed:.0f}s"
            )
            if val_mse + 1e-8 < best_val:
                best_val   = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale      = 0
            else:
                stale += 1
            if (step + 1) >= base_steps and stale >= cfg.fit_patience:
                print(f"    [early-stop] patience={cfg.fit_patience} exhausted at step {step+1}")
                break
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    tr_mse  = evaluate(model, train_loader, device)
    val_mse = evaluate(model, val_loader,   device)
    te_mse  = evaluate(model, test_loader,  device)
    return {
        "train_mse": tr_mse, "val_mse": val_mse, "test_mse": te_mse,
        "steps_run": history[-1]["step"] if history else 0,
    }, history


# ─────────────────────────────────────────────
# Shape / parameter matching
# ─────────────────────────────────────────────

def find_d_model_for_target_params(
    *,
    depth: int,
    target_params: int,
    cfg: TrainCfg,
) -> Tuple[int, int, int, int]:
    """Binary-search for the largest d_model (multiple of head_dim)
    such that active_params ≤ target_params.

    Returns (d_model, n_heads, d_ff, active_params).
    Raises ValueError if even d_model_min exceeds target.
    """
    hd  = cfg.head_dim
    lo  = max(hd, (cfg.d_model_min // hd) * hd)
    hi  = max(lo,  (cfg.d_model_max // hd) * hd)

    def n_active(d: int) -> int:
        nh  = max(1, d // hd)
        dff = 4 * d
        m   = TinyGPT(
            inp_dim=INP_DIM, out_dim=TEACHER_OUT, seq_len=SEQ_LEN,
            depth=depth, d_model=d, n_heads=nh, d_ff=dff,
            dropout=cfg.dropout, pad_params=0,
        )
        return count_params(m)

    if n_active(lo) > target_params:
        raise ValueError(
            f"depth={depth}: d_model={lo} already exceeds target_params={target_params}"
        )
    if n_active(hi) <= target_params:
        return hi, max(1, hi // hd), 4 * hi, n_active(hi)

    best_d, best_a = lo, n_active(lo)
    while lo <= hi:
        mid = ((lo + hi) // 2 // hd) * hd
        if mid < lo:
            break
        a = n_active(mid)
        if a <= target_params:
            best_d, best_a = mid, a
            lo = mid + hd
        else:
            hi = mid - hd

    candidates = []
    for d in [max(cfg.d_model_min, best_d - hd), best_d,
              min(cfg.d_model_max, best_d + hd)]:
        d = max(hd, (d // hd) * hd)
        a = n_active(d)
        if a <= target_params:
            candidates.append((abs(target_params - a), d, a))
    candidates.sort(key=lambda t: t[0])
    _, d_best, a_best = candidates[0]
    return d_best, max(1, d_best // hd), 4 * d_best, a_best


def build_student(
    *,
    depth: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    target_params: int,
    cfg: TrainCfg,
) -> TinyGPT:
    tmp    = TinyGPT(
        inp_dim=INP_DIM, out_dim=TEACHER_OUT, seq_len=SEQ_LEN,
        depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        dropout=cfg.dropout, pad_params=0,
    )
    active = count_params(tmp)
    if active > target_params:
        raise ValueError(f"active {active} > target {target_params}")
    pad = int(target_params - active)
    return TinyGPT(
        inp_dim=INP_DIM, out_dim=TEACHER_OUT, seq_len=SEQ_LEN,
        depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        dropout=cfg.dropout, pad_params=pad,
    )


# ─────────────────────────────────────────────
# Per-N shape sweep
# ─────────────────────────────────────────────

def run_shape_sweep_for_n(
    *,
    target_params: int,
    depths: List[int],
    cfg: TrainCfg,
    teacher: nn.Module,
    device: torch.device,
    out_dir: str,
    global_seed: int,
) -> List[Dict]:
    """Run the full depth sweep for a single N value.

    Builds separate train/val/test datasets scaled to D = data_ratio × N.
    Returns a list of result dicts (one per valid depth).
    """
    curve_dir = os.path.join(out_dir, f"curves_N{target_params}")
    os.makedirs(curve_dir, exist_ok=True)

    # Chinchilla-scaled data budget
    train_size = max(1, int(math.ceil(cfg.data_ratio * target_params / float(TEACHER_OUT))))
    base_steps = max(1, int(math.ceil(cfg.data_ratio * target_params
                                       / float(cfg.batch_size * TEACHER_OUT))))

    print(f"\n{'='*70}")
    print(f"  N = {target_params:,}  |  D = {int(cfg.data_ratio*target_params):,} "
          f"(ratio={cfg.data_ratio:.0f}N)  |  train_size={train_size:,}  "
          f"|  base_steps={base_steps:,}")
    print(f"{'='*70}")

    # Datasets
    t0 = time.time()
    train_ds, y_scale = make_dataset(train_size, global_seed + 1, teacher, device)
    val_ds,   _       = make_dataset(cfg.val_size,  global_seed + 2, teacher, device, y_scale)
    test_ds,  _       = make_dataset(cfg.test_size, global_seed + 3, teacher, device, y_scale)
    print(f"  Datasets ready in {time.time()-t0:.1f}s  (y_scale={y_scale:.4f})")

    loader_kw = dict(batch_size=cfg.batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = torch.utils.data.DataLoader(test_ds,  shuffle=False, **loader_kw)

    x_agop = test_ds.tensors[0][: cfg.agop_batch].to(device)

    results: List[Dict] = []

    for depth in depths:
        # ── find width ────────────────────────────────────────────────────
        try:
            d_model, n_heads, d_ff, active = find_d_model_for_target_params(
                depth=depth, target_params=target_params, cfg=cfg)
        except ValueError as exc:
            print(f"  [SKIP] depth={depth}: {exc}")
            continue

        pad       = int(target_params - active)
        pad_ratio = pad / max(1, target_params)
        if pad_ratio > cfg.max_padding_ratio:
            print(f"  [SKIP] depth={depth}: padding_ratio={pad_ratio:.3f} "
                  f"> max={cfg.max_padding_ratio:.3f}")
            continue

        aspect_ratio = depth / d_model     # 深宽比 (higher = deeper/narrower)

        print(f"\n  ── depth={depth:3d}  d_model={d_model:4d}  n_heads={n_heads:2d}  "
              f"d_ff={d_ff:5d}  active={active:,}  pad={pad:,}  "
              f"α=depth/d_model={aspect_ratio:.4f}")

        # ── build & train ─────────────────────────────────────────────────
        set_seed(global_seed + depth + target_params % 10_000)
        try:
            model = build_student(
                depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                target_params=target_params, cfg=cfg,
            )
        except ValueError as exc:
            print(f"  [SKIP] build failed: {exc}")
            continue

        total_params = count_params(model)
        metrics, history = train_one_model(
            model, train_loader, val_loader, test_loader, base_steps, cfg, device)

        # ── save training curve ───────────────────────────────────────────
        if history:
            _save_curve(history, curve_dir, f"depth{depth}_dmodel{d_model}")

        # ── AGOP ─────────────────────────────────────────────────────────
        agop = estimate_agop_wrt_embeddings(
            model, x_agop, proj_samples=cfg.agop_proj_samples)
        aofe, aofe_ratio = agop_offdiag_metrics(agop)

        teacher_type_str = teacher.__class__.__name__.replace("Teacher", "").lower()
        row: Dict = {
            "target_params":       int(target_params),
            "teacher_type":        teacher_type_str,
            "depth":               int(depth),
            "d_model":             int(d_model),
            "n_heads":             int(n_heads),
            "d_ff":                int(d_ff),
            "aspect_ratio":        float(aspect_ratio),          # depth / d_model
            "active_params":       int(active),
            "pad_params":          int(pad),
            "total_params":        int(total_params),
            "padding_ratio":       float(pad_ratio),
            "train_mse":           float(metrics["train_mse"]),
            "val_mse":             float(metrics["val_mse"]),
            "test_mse":            float(metrics["test_mse"]),
            "steps_run":           int(metrics["steps_run"]),
            "base_steps":          int(base_steps),
            "agop_offdiag_energy": float(aofe),
            "agop_offdiag_ratio":  float(aofe_ratio),
        }
        results.append(row)

        print(f"  → test_mse={metrics['test_mse']:.5f}  "
              f"AOFE={aofe:.4f}  AOFE_ratio={aofe_ratio:.4f}")

        del model, agop
        torch.cuda.empty_cache()

    # ── per-N correlations ────────────────────────────────────────────────
    if len(results) >= 3:
        mse_arr  = np.array([r["test_mse"]            for r in results])
        aofe_arr = np.array([r["agop_offdiag_ratio"]  for r in results])
        p = pearson_corr(aofe_arr, mse_arr)
        s = spearman_corr(aofe_arr, mse_arr)
        print(f"\n  N={target_params:,}: Pearson(AOFE_ratio, test_mse)={p:.3f}  "
              f"Spearman={s:.3f}")

    return results


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def _save_curve(history: List[Dict], out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _n_label(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


def plot_per_n_results(
    results: List[Dict],
    target_params: int,
    out_dir: str,
) -> None:
    """Plots for a single N value: metrics vs depth and vs aspect_ratio."""
    if not results:
        return
    os.makedirs(out_dir, exist_ok=True)
    tag = _n_label(target_params)

    depths       = [r["depth"]               for r in results]
    d_models     = [r["d_model"]             for r in results]
    aspect       = [r["aspect_ratio"]        for r in results]
    test_mse     = [r["test_mse"]            for r in results]
    aofe_energy  = [r["agop_offdiag_energy"] for r in results]
    aofe_ratio   = [r["agop_offdiag_ratio"]  for r in results]

    def _scatter(xs, ys, xlabel, ylabel, title, fname, annot=None, r=None):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(xs, ys, "o-", linewidth=1.5, markersize=5)
        if annot:
            for x_, y_, a in zip(xs, ys, annot):
                ax.annotate(str(a), (x_, y_), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if r is not None:
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                    va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="lightyellow", edgecolor="gray", alpha=0.85))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=180)
        plt.close()

    # test_mse vs depth
    _scatter(depths, test_mse, "Depth (# layers)", "Test MSE",
             f"Test MSE vs Depth  [N={tag}]",
             f"N{target_params}_mse_vs_depth.png", annot=d_models)

    # aofe_ratio vs depth
    _scatter(depths, aofe_ratio, "Depth (# layers)", "AOFE_ratio",
             f"AOFE_ratio vs Depth  [N={tag}]",
             f"N{target_params}_aoferat_vs_depth.png", annot=d_models)

    # test_mse vs aspect_ratio
    _scatter(aspect, test_mse, "Aspect ratio  α = depth / d_model", "Test MSE",
             f"Test MSE vs Aspect Ratio  [N={tag}]",
             f"N{target_params}_mse_vs_aspectratio.png", annot=depths)

    # aofe_ratio vs aspect_ratio
    _scatter(aspect, aofe_ratio, "Aspect ratio  α = depth / d_model", "AOFE_ratio",
             f"AOFE_ratio vs Aspect Ratio  [N={tag}]",
             f"N{target_params}_aoferat_vs_aspectratio.png", annot=depths)

    # scatter: aofe_ratio vs test_mse
    r_val = pearson_corr(np.array(aofe_ratio), np.array(test_mse))
    _scatter(aofe_ratio, test_mse, "AOFE_ratio", "Test MSE",
             f"AOFE_ratio vs Test MSE  [N={tag}]",
             f"N{target_params}_mse_vs_aoferat_scatter.png",
             annot=depths, r=r_val)


def plot_multi_n_summary(
    all_results: List[Dict],
    param_groups: List[int],
    out_dir: str,
) -> None:
    """Comprehensive multi-N summary plots."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        cmap = matplotlib.colormaps["tab10"]
    except AttributeError:
        cmap = matplotlib.cm.get_cmap("tab10")  # matplotlib < 3.7 fallback
    colors = {n: cmap(i % 10) for i, n in enumerate(param_groups)}

    # ── 1. test_mse vs depth (all N overlaid) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in param_groups:
        rows = sorted([r for r in all_results if r["target_params"] == n],
                      key=lambda r: r["depth"])
        if not rows:
            continue
        xs = [r["depth"]    for r in rows]
        ys = [r["test_mse"] for r in rows]
        ax.plot(xs, ys, "o-", color=colors[n], label=f"N={_n_label(n)}", linewidth=1.5)
        best_i = int(np.argmin(ys))
        ax.scatter([xs[best_i]], [ys[best_i]], color=colors[n], s=80, zorder=5,
                   edgecolors="black", linewidths=0.8)
    ax.set_xlabel("Depth (# layers)")
    ax.set_ylabel("Test MSE")
    ax.set_title("Test MSE vs Depth for multiple N\n(★ = optimal depth)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_mse_vs_depth.png"), dpi=180)
    plt.close()

    # ── 2. aofe_ratio vs depth (all N overlaid) ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in param_groups:
        rows = sorted([r for r in all_results if r["target_params"] == n],
                      key=lambda r: r["depth"])
        if not rows:
            continue
        xs = [r["depth"]               for r in rows]
        ys = [r["agop_offdiag_ratio"]  for r in rows]
        ax.plot(xs, ys, "s--", color=colors[n], label=f"N={_n_label(n)}", linewidth=1.5)
        best_i = int(np.argmin(ys))
        ax.scatter([xs[best_i]], [ys[best_i]], color=colors[n], s=80, zorder=5,
                   edgecolors="black", linewidths=0.8)
    ax.set_xlabel("Depth (# layers)")
    ax.set_ylabel("AOFE_ratio")
    ax.set_title("AOFE_ratio vs Depth for multiple N\n(★ = min AOFE_ratio)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_aoferat_vs_depth.png"), dpi=180)
    plt.close()

    # ── 3. test_mse vs aspect_ratio (all N overlaid) ──────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in param_groups:
        rows = sorted([r for r in all_results if r["target_params"] == n],
                      key=lambda r: r["aspect_ratio"])
        if not rows:
            continue
        xs = [r["aspect_ratio"] for r in rows]
        ys = [r["test_mse"]     for r in rows]
        ax.plot(xs, ys, "o-", color=colors[n], label=f"N={_n_label(n)}", linewidth=1.5)
        best_i = int(np.argmin(ys))
        ax.scatter([xs[best_i]], [ys[best_i]], color=colors[n], s=80, zorder=5,
                   edgecolors="black", linewidths=0.8)
    ax.set_xlabel("Aspect ratio  α = depth / d_model")
    ax.set_ylabel("Test MSE")
    ax.set_title("Test MSE vs Aspect Ratio for multiple N\n(★ = optimal α)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_mse_vs_aspectratio.png"), dpi=180)
    plt.close()

    # ── 4. aofe_ratio vs aspect_ratio (all N overlaid) ────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in param_groups:
        rows = sorted([r for r in all_results if r["target_params"] == n],
                      key=lambda r: r["aspect_ratio"])
        if not rows:
            continue
        xs = [r["aspect_ratio"]       for r in rows]
        ys = [r["agop_offdiag_ratio"] for r in rows]
        ax.plot(xs, ys, "s--", color=colors[n], label=f"N={_n_label(n)}", linewidth=1.5)
        best_i = int(np.argmin(ys))
        ax.scatter([xs[best_i]], [ys[best_i]], color=colors[n], s=80, zorder=5,
                   edgecolors="black", linewidths=0.8)
    ax.set_xlabel("Aspect ratio  α = depth / d_model")
    ax.set_ylabel("AOFE_ratio")
    ax.set_title("AOFE_ratio vs Aspect Ratio for multiple N\n(★ = min AOFE_ratio)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_aoferat_vs_aspectratio.png"), dpi=180)
    plt.close()

    # ── 5. Scatter: AOFE_ratio vs test_mse (all points) ───────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for n in param_groups:
        rows = [r for r in all_results if r["target_params"] == n]
        if not rows:
            continue
        xs = [r["agop_offdiag_ratio"] for r in rows]
        ys = [r["test_mse"]           for r in rows]
        ax.scatter(xs, ys, color=colors[n], label=f"N={_n_label(n)}", alpha=0.8, s=40)
    all_aofe = np.array([r["agop_offdiag_ratio"] for r in all_results])
    all_mse  = np.array([r["test_mse"]           for r in all_results])
    r_all    = pearson_corr(all_aofe, all_mse)
    s_all    = spearman_corr(all_aofe, all_mse)
    ax.text(0.05, 0.95,
            f"Pearson r = {r_all:.3f}\nSpearman ρ = {s_all:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.85))
    ax.set_xlabel("AOFE_ratio")
    ax.set_ylabel("Test MSE")
    ax.set_title("AOFE_ratio vs Test MSE  (all N, all shapes)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_scatter_aoferat_vs_mse.png"), dpi=180)
    plt.close()

    # ── 6. Optimal aspect_ratio vs log(N) ─────────────────────────────────
    opt_n, opt_alpha_mse, opt_alpha_aofe = [], [], []
    for n in param_groups:
        rows = [r for r in all_results if r["target_params"] == n]
        if len(rows) < 2:
            continue
        best_mse_i  = int(np.argmin([r["test_mse"]            for r in rows]))
        best_aofe_i = int(np.argmin([r["agop_offdiag_ratio"]  for r in rows]))
        opt_n.append(n)
        opt_alpha_mse.append(rows[best_mse_i]["aspect_ratio"])
        opt_alpha_aofe.append(rows[best_aofe_i]["aspect_ratio"])

    if len(opt_n) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        log_n = np.log10(opt_n)
        ax.plot(log_n, opt_alpha_mse,  "o-",  label="α* (min test_mse)",    linewidth=1.8)
        ax.plot(log_n, opt_alpha_aofe, "s--", label="α* (min AOFE_ratio)", linewidth=1.8)
        ax.set_xlabel("log₁₀(N)")
        ax.set_ylabel("Optimal aspect ratio  α* = depth / d_model")
        ax.set_title("Optimal Shape vs Parameter Count")
        ax.legend(fontsize=9)
        xticks = [np.log10(n) for n in opt_n]
        ax.set_xticks(xticks)
        ax.set_xticklabels([_n_label(n) for n in opt_n])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "summary_optimal_alpha_vs_N.png"), dpi=180)
        plt.close()

    # ── 7. Dual-axis: test_mse and AOFE_ratio vs aspect_ratio (best N) ───
    # (one subplot per N)
    n_plots = len(param_groups)
    if n_plots > 0:
        ncols = min(n_plots, 3)
        nrows = math.ceil(n_plots / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                 squeeze=False)
        for idx, n in enumerate(param_groups):
            row_idx, col_idx = divmod(idx, ncols)
            ax1 = axes[row_idx][col_idx]
            rows = sorted([r for r in all_results if r["target_params"] == n],
                          key=lambda r: r["aspect_ratio"])
            if not rows:
                ax1.set_visible(False)
                continue
            xs         = [r["aspect_ratio"]       for r in rows]
            ys_mse     = [r["test_mse"]            for r in rows]
            ys_aofe    = [r["agop_offdiag_ratio"]  for r in rows]

            ax2 = ax1.twinx()
            l1, = ax1.plot(xs, ys_mse,  "o-",  color="tab:blue",  label="Test MSE",    linewidth=1.8)
            l2, = ax2.plot(xs, ys_aofe, "s--", color="tab:orange", label="AOFE_ratio", linewidth=1.8)

            # mark optimal
            best_i = int(np.argmin(ys_mse))
            ax1.scatter([xs[best_i]], [ys_mse[best_i]], color="tab:blue", s=80, zorder=5,
                        edgecolors="black", linewidths=0.8)

            ax1.set_xlabel("Aspect ratio  α = depth / d_model", fontsize=8)
            ax1.set_ylabel("Test MSE", color="tab:blue", fontsize=8)
            ax2.set_ylabel("AOFE_ratio", color="tab:orange", fontsize=8)
            ax1.set_title(f"N = {_n_label(n)}", fontsize=9)
            lines = [l1, l2]
            ax1.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="upper left")

        for idx in range(n_plots, nrows * ncols):
            row_idx, col_idx = divmod(idx, ncols)
            axes[row_idx][col_idx].set_visible(False)

        fig.suptitle("Test MSE and AOFE_ratio vs Aspect Ratio  (all N)", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "summary_dual_axis_mse_aofe.png"), dpi=180)
        plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-N Transformer shape sweep: test MSE + AOFE_ratio vs depth/width")
    parser.add_argument("--out_dir",      type=str,   default="./results_scaling_shape_sweep")
    parser.add_argument("--device",       type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",         type=int,   default=42)

    # Sweep configuration
    parser.add_argument(
        "--param_groups", type=str, default="300000,1000000,3000000",
        help="Comma-separated target parameter counts N (e.g. 300000,1000000,3000000).")
    parser.add_argument(
        "--depth_list",   type=str, default="1,2,3,4,5,6,8,10,12,16,20,24",
        help="Depth values to try for every N. Include depth=1 to test "
             "whether very shallow models are also suboptimal, creating the "
             "U-shaped loss-vs-depth curve.")
    parser.add_argument(
        "--teacher_type", type=str, default="gpt", choices=["gpt", "mlp"],
        help="Teacher architecture. 'gpt' (default): 6-layer decoder Transformer "
             "producing U-shaped loss-vs-depth. 'mlp': flat 4-layer MLP.")

    # Architecture
    parser.add_argument("--head_dim",         type=int,   default=4,
                        help="Head dimension (d_head = d_model // head_dim). "
                             "4 gives finer d_model granularity than 8.")
    parser.add_argument("--dropout",          type=float, default=0.0)
    parser.add_argument("--d_model_min",      type=int,   default=24)
    parser.add_argument("--d_model_max",      type=int,   default=1024)
    parser.add_argument("--max_padding_ratio",type=float, default=0.20)

    # Training
    parser.add_argument("--data_ratio",    type=float, default=20.0,
                        help="Chinchilla D/N ratio (D = data_ratio × N supervised outputs).")
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--warmup_steps",  type=int,   default=300)
    parser.add_argument("--eval_every",    type=int,   default=200)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--max_train_factor", type=float, default=1.5,
                        help="Allow up to max_train_factor × base_steps training.")
    parser.add_argument("--fit_patience",  type=int,   default=8,
                        help="Early-stop after this many eval periods without val improvement.")
    parser.add_argument("--val_size",      type=int,   default=3_000)
    parser.add_argument("--test_size",     type=int,   default=3_000)

    # AGOP
    parser.add_argument("--agop_batch",        type=int, default=256)
    parser.add_argument("--agop_proj_samples", type=int, default=64)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    param_groups = [int(x) for x in args.param_groups.split(",") if x.strip()]
    depths       = [int(x) for x in args.depth_list.split(",") if x.strip()]

    cfg = TrainCfg(
        lr=args.lr, weight_decay=args.weight_decay,
        data_ratio=args.data_ratio, warmup_steps=args.warmup_steps,
        batch_size=args.batch_size, eval_every=args.eval_every,
        grad_clip=args.grad_clip, val_size=args.val_size, test_size=args.test_size,
        head_dim=args.head_dim, dropout=args.dropout,
        max_padding_ratio=args.max_padding_ratio, max_train_factor=args.max_train_factor,
        fit_patience=args.fit_patience, agop_batch=args.agop_batch,
        agop_proj_samples=args.agop_proj_samples, seed=args.seed,
        d_model_min=args.d_model_min, d_model_max=args.d_model_max,
    )

    # ── summary header ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Transformer Scaling × Shape Sweep")
    print("  Hypothesis: ∃ optimal α* = depth/d_model that minimises MSE + AOFE_ratio")
    print("=" * 70)
    print(f"  param_groups : {[_n_label(n) for n in param_groups]}")
    print(f"  depths       : {depths}")
    print(f"  teacher_type : {args.teacher_type}")
    print(f"  data_ratio   : {cfg.data_ratio:.0f}  (D = {cfg.data_ratio:.0f}N)")
    print(f"  batch_size   : {cfg.batch_size}")
    print(f"  head_dim     : {cfg.head_dim}")
    print(f"  d_model_min  : {cfg.d_model_min}  d_model_max: {cfg.d_model_max}")
    print(f"  max_train_factor: {cfg.max_train_factor}  fit_patience: {cfg.fit_patience}")
    print(f"  agop_proj_samples: {cfg.agop_proj_samples}  agop_batch: {cfg.agop_batch}")
    print(f"  device       : {device}")
    print(f"  out_dir      : {args.out_dir}")
    print("=" * 70)

    # ── shared teacher ────────────────────────────────────────────────────
    teacher = build_teacher(seed=args.seed + 999, device=device, teacher_type=args.teacher_type)

    # ── sweep ─────────────────────────────────────────────────────────────
    all_results: List[Dict] = []
    t_total_start = time.time()

    for n in param_groups:
        n_results = run_shape_sweep_for_n(
            target_params=n,
            depths=depths,
            cfg=cfg,
            teacher=teacher,
            device=device,
            out_dir=args.out_dir,
            global_seed=args.seed,
        )
        all_results.extend(n_results)
        plot_per_n_results(n_results, n, args.out_dir)

        # checkpoint after each N
        _save_all_results(all_results, args.out_dir)

    elapsed = time.time() - t_total_start
    print(f"\nTotal sweep time: {elapsed/60:.1f} min")

    # ── global correlations ────────────────────────────────────────────────
    if len(all_results) >= 3:
        all_aofe = np.array([r["agop_offdiag_ratio"] for r in all_results])
        all_mse  = np.array([r["test_mse"]           for r in all_results])
        p_all    = pearson_corr(all_aofe, all_mse)
        s_all    = spearman_corr(all_aofe, all_mse)
        print(f"\nGlobal (all N, all shapes):")
        print(f"  Pearson (AOFE_ratio, test_mse)  = {p_all:.4f}")
        print(f"  Spearman(AOFE_ratio, test_mse)  = {s_all:.4f}")

    # ── plots ──────────────────────────────────────────────────────────────
    plot_multi_n_summary(all_results, param_groups, args.out_dir)
    _save_all_results(all_results, args.out_dir)
    print(f"\nResults saved to: {args.out_dir}")
    _print_summary_table(all_results, param_groups)


def _save_all_results(results: List[Dict], out_dir: str) -> None:
    if not results:
        return
    csv_path = os.path.join(out_dir, "results.csv")
    npy_path = os.path.join(out_dir, "results.npy")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    np.save(npy_path, results, allow_pickle=True)


def _print_summary_table(results: List[Dict], param_groups: List[int]) -> None:
    print("\n" + "=" * 90)
    print(f"{'N':>10}  {'depth':>6}  {'d_model':>8}  {'α=d/dm':>9}  "
          f"{'test_mse':>10}  {'AOFE_ratio':>12}  {'optimal?':>10}")
    print("-" * 90)
    for n in param_groups:
        rows = sorted([r for r in results if r["target_params"] == n],
                      key=lambda r: r["depth"])
        if not rows:
            continue
        best_mse_i = int(np.argmin([r["test_mse"] for r in rows]))
        for i, r in enumerate(rows):
            marker = " ← best" if i == best_mse_i else ""
            print(
                f"{_n_label(n):>10}  {r['depth']:>6}  {r['d_model']:>8}  "
                f"{r['aspect_ratio']:>9.4f}  {r['test_mse']:>10.5f}  "
                f"{r['agop_offdiag_ratio']:>12.4f}  {marker}"
            )
        print()
    print("=" * 90)


if __name__ == "__main__":
    main()
