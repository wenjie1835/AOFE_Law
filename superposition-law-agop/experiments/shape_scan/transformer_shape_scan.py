"""
Tiny-GPT depth/width scan with fixed parameter budget and AOFE metrics.

Goal
----
Scan n_layer = 2..12 while keeping each model close to a fixed parameter
budget (default: 3M parameters). Train on WikiText with the Chinchilla-style
token budget D = 20 * N_params, then record final validation loss, AOFE, and
AOFE-ratio. Finally, compute correlations between loss and the two AOFE metrics.

AOFE convention
---------------
This follows the naming/measurement convention in data_scaling.py:

    AOFE       = off-diagonal Frobenius energy of an AGOP-like matrix
    AOFE-ratio = AOFE / total Frobenius energy

For GPT, tokens are discrete, so the AGOP-like matrix is computed over the
input embedding channel dimension. For a validation batch, we take gradients of
the summed language-modeling loss with respect to the token embeddings, average
g g^T over token positions, and apply the same off-diagonal energy functions
used by data_scaling.py.

Examples
--------
Full default experiment:

    python3 tiny_gpt_depth_aofe_scan.py

Quick smoke test:

    python3 tiny_gpt_depth_aofe_scan.py --dataset wikitext-2 --layers 2 3 --train-tokens 200000 --max-steps 20

Inspect the parameter-matched architectures without training:

    python3 tiny_gpt_depth_aofe_scan.py --dry-run-configs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Same AOFE/AOFE-ratio convention as data_scaling.py, copied here to avoid
# importing the plotting stack when running long GPT experiments.
@torch.no_grad()
def offdiag_energy_from_matrix(mat: torch.Tensor) -> float:
    diag = torch.diagonal(mat)
    off = mat - torch.diag(diag)
    return float((off ** 2).sum().item())


@torch.no_grad()
def offdiag_energy_ratio_from_matrix(mat: torch.Tensor, eps: float = 1e-12) -> float:
    diag = torch.diagonal(mat)
    off = mat - torch.diag(diag)
    num = (off ** 2).sum()
    den = (mat ** 2).sum().clamp_min(eps)
    return float((num / den).item())


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    def rankdata(a: np.ndarray) -> np.ndarray:
        temp = a.argsort()
        ranks = np.empty_like(temp, dtype=np.float64)
        ranks[temp] = np.arange(len(a), dtype=np.float64)
        return ranks

    return pearsonr(rankdata(np.asarray(x)), rankdata(np.asarray(y)))


# Legacy MetaMind URLs return HTTP 301 + S3 PermanentRedirect XML without a Location header;
# ``urllib.request.urlretrieve`` then fails. Prefer ``prepare_wikitext_bytes`` HuggingFace path.
WIKITEXT_URLS = {
    "wikitext-103": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
    "wikitext-2": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
}

WIKITEXT_HF_CONFIG = {
    "wikitext-103": "wikitext-103-v1",
    "wikitext-2": "wikitext-2-v1",
}


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, width = x.size()
        q, k, v = self.c_attn(x).split(width, dim=2)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, width)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Standard GPT weight tying.
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: Optional[torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        tok_emb_override: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if tok_emb_override is None:
            if idx is None:
                raise ValueError("idx is required unless tok_emb_override is provided")
            tok_emb = self.wte(idx)
        else:
            tok_emb = tok_emb_override

        _, seq_len, _ = tok_emb.size()
        if seq_len > self.cfg.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(0, seq_len, dtype=torch.long, device=tok_emb.device)
        x = self.drop(tok_emb + self.wpe(pos)[None, :, :])
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction=reduction,
            )
        return logits, loss

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _s3_redirect_endpoint_from_error(exc: urllib.error.HTTPError) -> Optional[str]:
    """Parse S3 PermanentRedirect XML body for a virtual-hosted endpoint host."""
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    if "PermanentRedirect" not in body or "<Endpoint>" not in body:
        return None
    try:
        root = ET.fromstring(body)
        el = root.find("Endpoint")
        if el is not None and el.text:
            return el.text.strip()
    except ET.ParseError:
        pass
    return None


def download_with_progress(url: str, dest: Path) -> None:
    """
    Stream a URL to disk with a simple progress line.
    Handles S3 ``PermanentRedirect`` (301 + XML endpoint) by retrying on the suggested host.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    max_redirects = 4
    current = url
    for _ in range(max_redirects):
        req = urllib.request.Request(
            current,
            headers={"User-Agent": "Mozilla/5.0 (compatible; transformer_shape_scan/1.0)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                done = 0
                block = 1024 * 256
                with dest.open("wb") as out:
                    while True:
                        chunk = resp.read(block)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total > 0:
                            pct = 100.0 * min(done, total) / total
                            print(f"\rDownloading {dest.name}: {pct:5.1f}%", end="", flush=True)
                print()
                return
        except urllib.error.HTTPError as e:
            host = _s3_redirect_endpoint_from_error(e)
            if host and e.code in (301, 307):
                path = urllib.parse.urlparse(current).path
                current = f"https://{host}{path}"
                print(f"  [S3 redirect] retrying: {current}")
                continue
            raise
    raise RuntimeError(f"Too many S3 redirects while downloading {url}")


def _materialize_wikitext_from_hf(dataset: str, extracted_dir: Path) -> None:
    """Write ``wiki.{train,valid}.tokens`` using HuggingFace ``Salesforce/wikitext`` (recommended)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the `datasets` package (see repo requirements.txt) or use a pre-downloaded WikiText zip."
        ) from exc

    if dataset not in WIKITEXT_HF_CONFIG:
        raise ValueError(f"Unknown dataset {dataset!r}")
    config = WIKITEXT_HF_CONFIG[dataset]
    print(f"Loading {dataset} via HuggingFace (Salesforce/wikitext, {config}) …")
    ds = load_dataset("Salesforce/wikitext", config)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    train_text = "\n".join(ds["train"]["text"])
    val_key = "validation" if "validation" in ds else "valid"
    valid_text = "\n".join(ds[val_key]["text"])
    (extracted_dir / "wiki.train.tokens").write_text(train_text, encoding="utf-8")
    (extracted_dir / "wiki.valid.tokens").write_text(valid_text, encoding="utf-8")


def prepare_wikitext_bytes(data_dir: Path, dataset: str) -> Tuple[Path, Path]:
    if dataset not in WIKITEXT_URLS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose one of {sorted(WIKITEXT_URLS)}")

    raw_dir = data_dir / dataset
    train_bin = raw_dir / "train_uint16.bin"
    valid_bin = raw_dir / "valid_uint16.bin"
    if train_bin.exists() and valid_bin.exists():
        return train_bin, valid_bin

    zip_path = data_dir / f"{dataset}-v1.zip"
    extracted_dir = raw_dir / dataset
    train_txt = extracted_dir / "wiki.train.tokens"
    valid_txt = extracted_dir / "wiki.valid.tokens"

    if not train_txt.exists() or not valid_txt.exists():
        hf_ok = False
        try:
            _materialize_wikitext_from_hf(dataset, extracted_dir)
            hf_ok = True
        except Exception as exc:
            print(f"[warn] HuggingFace WikiText path failed: {exc}")
        if not hf_ok:
            if not zip_path.exists():
                print(f"Downloading {dataset} from {WIKITEXT_URLS[dataset]}")
                download_with_progress(WIKITEXT_URLS[dataset], zip_path)
            print(f"Extracting {zip_path}")
            raw_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(raw_dir)

    if not train_txt.exists() or not valid_txt.exists():
        raise FileNotFoundError(f"Could not find WikiText files under {extracted_dir}")

    for txt_path, bin_path in [(train_txt, train_bin), (valid_txt, valid_bin)]:
        print(f"Byte-tokenizing {txt_path.name} -> {bin_path.name}")
        text = txt_path.read_text(encoding="utf-8")
        arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16)
        arr.tofile(bin_path)

    return train_bin, valid_bin


def open_uint16_memmap(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(f"Dataset has {len(data)} byte tokens, smaller than block_size={block_size}")
    starts = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([np.asarray(data[i : i + block_size], dtype=np.int64) for i in starts])
    y = np.stack([np.asarray(data[i + 1 : i + 1 + block_size], dtype=np.int64) for i in starts])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def count_params_for_config(cfg: GPTConfig) -> int:
    with torch.device("meta"):
        model = TinyGPT(cfg)
    return model.parameter_count()


def divisors(n: int) -> Iterable[int]:
    for i in range(1, n + 1):
        if n % i == 0:
            yield i


def make_config_for_param_budget(
    n_layer: int,
    target_params: int,
    block_size: int,
    vocab_size: int,
    dropout: float,
    bias: bool,
) -> Tuple[GPTConfig, int]:
    best: Optional[Tuple[int, GPTConfig, int]] = None
    for n_embd in range(32, 1025, 8):
        valid_heads = [
            h for h in divisors(n_embd)
            if 1 <= h <= 16 and 16 <= (n_embd // h) <= 96
        ]
        if not valid_heads:
            continue
        n_head = min(valid_heads, key=lambda h: (abs((n_embd // h) - 64), h))
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
        )
        params = count_params_for_config(cfg)
        err = abs(params - target_params)
        if best is None or err < best[0]:
            best = (err, cfg, params)

    if best is None:
        raise RuntimeError("No valid GPT config candidates found")
    return best[1], best[2]


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y, reduction="mean")
        if loss is None:
            raise RuntimeError("loss unexpectedly missing")
        losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses))


def configure_optimizer(
    model: TinyGPT,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
) -> torch.optim.Optimizer:
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and not name.endswith("wte.weight"):
            decay_params.append(param)
        else:
            nodecay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )


def cosine_lr(step: int, max_steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def train_one_model(
    model: TinyGPT,
    train_data: np.ndarray,
    valid_data: np.ndarray,
    train_tokens: int,
    batch_size: int,
    grad_accum: int,
    eval_interval: int,
    eval_iters: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    warmup_steps: int,
    max_steps: Optional[int],
    device: torch.device,
    compile_model: bool,
) -> Dict[str, float]:
    model.to(device)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    tokens_per_step = batch_size * model.cfg.block_size * grad_accum
    planned_steps = math.ceil(train_tokens / tokens_per_step)
    steps = min(planned_steps, max_steps) if max_steps is not None else planned_steps
    opt = configure_optimizer(model, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start = time.time()
    last_train_loss = float("nan")
    best_val_loss = float("inf")

    print(
        f"  training steps={steps} planned_steps={planned_steps} "
        f"tokens_per_step={tokens_per_step}"
    )

    for step in range(steps):
        lr_now = cosine_lr(step, steps, lr, min_lr, warmup_steps)
        for group in opt.param_groups:
            group["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(train_data, batch_size, model.cfg.block_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = model(x, y, reduction="mean")
                if loss is None:
                    raise RuntimeError("loss unexpectedly missing")
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            loss_accum += float(loss.detach().cpu()) * grad_accum

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        last_train_loss = loss_accum

        if step == 0 or (step + 1) % eval_interval == 0 or step == steps - 1:
            val_loss = estimate_loss(model, valid_data, batch_size, model.cfg.block_size, eval_iters, device)
            best_val_loss = min(best_val_loss, val_loss)
            elapsed = time.time() - start
            print(
                f"    step={step + 1:5d}/{steps} "
                f"train_loss={last_train_loss:.4f} val_loss={val_loss:.4f} "
                f"lr={lr_now:.2e} elapsed={elapsed / 60:.1f}m"
            )

    final_val_loss = estimate_loss(model, valid_data, batch_size, model.cfg.block_size, eval_iters, device)
    return {
        "steps": float(steps),
        "planned_steps": float(planned_steps),
        "tokens_per_step": float(tokens_per_step),
        "effective_train_tokens": float(steps * tokens_per_step),
        "final_train_loss": float(last_train_loss),
        "final_val_loss": float(final_val_loss),
        "best_val_loss": float(best_val_loss),
    }


def compute_embedding_channel_aofe(
    model: TinyGPT,
    valid_data: np.ndarray,
    batch_size: int,
    block_size: int,
    batches: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    agop = torch.zeros((model.cfg.n_embd, model.cfg.n_embd), device=device, dtype=torch.float32)
    total = 0
    for _ in range(batches):
        x, y = get_batch(valid_data, batch_size, block_size, device)
        tok_emb = model.wte(x).detach().requires_grad_(True)
        with torch.enable_grad():
            _, loss = model(None, y, tok_emb_override=tok_emb, reduction="sum")
            if loss is None:
                raise RuntimeError("loss unexpectedly missing")
            grad = torch.autograd.grad(loss, tok_emb, retain_graph=False, create_graph=False)[0]
        g = grad.reshape(-1, model.cfg.n_embd).to(torch.float32)
        agop += g.T @ g
        total += g.shape[0]
    agop /= max(1, total)
    agop = 0.5 * (agop + agop.T)
    return offdiag_energy_from_matrix(agop), offdiag_energy_ratio_from_matrix(agop)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_correlations(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    loss = np.asarray([float(r["final_val_loss"]) for r in rows], dtype=np.float64)
    aofe = np.asarray([float(r["aofe"]) for r in rows], dtype=np.float64)
    ratio = np.asarray([float(r["aofe_ratio"]) for r in rows], dtype=np.float64)
    return {
        "final_val_loss_vs_aofe": {
            "pearson": pearsonr(loss, aofe),
            "spearman": spearmanr(loss, aofe),
        },
        "final_val_loss_vs_aofe_ratio": {
            "pearson": pearsonr(loss, ratio),
            "spearman": spearmanr(loss, ratio),
        },
    }


def parse_layers(args_layers: List[int]) -> List[int]:
    if len(args_layers) == 1:
        return args_layers
    if len(args_layers) == 2:
        lo, hi = args_layers
        if lo > hi:
            raise ValueError("--layers START END requires START <= END")
        return list(range(lo, hi + 1))
    return args_layers


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Tiny-GPT depth at fixed params and compute AOFE metrics.")
    parser.add_argument("--target-params", type=int, default=3_000_000)
    parser.add_argument("--tokens-per-param", type=float, default=20.0)
    parser.add_argument("--train-tokens", type=int, default=None, help="Override Chinchilla token budget.")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 12], help="Either START END or explicit layer list.")
    parser.add_argument("--dataset", choices=sorted(WIKITEXT_URLS), default="wikitext-103")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--out-dir", type=Path, default=Path("./results_tiny_gpt_depth_aofe"))
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--aofe-batches", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Debug cap; using this breaks the Chinchilla budget.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--dry-run-configs", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    layers = parse_layers(args.layers)
    train_tokens = args.train_tokens
    if train_tokens is None:
        train_tokens = int(round(args.tokens_per_param * args.target_params))

    configs = []
    for n_layer in layers:
        cfg, params = make_config_for_param_budget(
            n_layer=n_layer,
            target_params=args.target_params,
            block_size=args.block_size,
            vocab_size=256,
            dropout=args.dropout,
            bias=True,
        )
        configs.append((cfg, params))

    print("Parameter-matched configs:")
    for cfg, params in configs:
        rel_err = 100.0 * (params - args.target_params) / args.target_params
        print(
            f"  L={cfg.n_layer:2d} C={cfg.n_embd:4d} H={cfg.n_head:2d} "
            f"params={params:,} ({rel_err:+.2f}%)"
        )
    print(f"Train token budget per model: {train_tokens:,} byte tokens")
    print(f"Device: {device}")
    if args.dry_run_configs:
        return

    train_bin, valid_bin = prepare_wikitext_bytes(args.data_dir, args.dataset)
    train_data = open_uint16_memmap(train_bin)
    valid_data = open_uint16_memmap(valid_bin)
    print(f"Loaded train tokens={len(train_data):,}, valid tokens={len(valid_data):,}")
    if len(train_data) < train_tokens:
        print(
            "Warning: train token budget exceeds unique dataset byte tokens; "
            "training will sample random windows with replacement."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"device_resolved": str(device)}, f, indent=2, default=str)

    for cfg, params in configs:
        print("\n" + "=" * 80)
        print(
            f"n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_head={cfg.n_head} "
            f"params={params:,}"
        )
        set_seed(args.seed + cfg.n_layer)
        model = TinyGPT(cfg)
        stats = train_one_model(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            train_tokens=train_tokens,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            device=device,
            compile_model=args.compile,
        )
        aofe, aofe_ratio = compute_embedding_channel_aofe(
            model=model,
            valid_data=valid_data,
            batch_size=args.batch_size,
            block_size=args.block_size,
            batches=args.aofe_batches,
            device=device,
        )
        row: Dict[str, object] = {
            "n_layer": cfg.n_layer,
            "n_embd": cfg.n_embd,
            "n_head": cfg.n_head,
            "head_dim": cfg.n_embd // cfg.n_head,
            "param_count": params,
            "param_error": params - args.target_params,
            "train_token_budget": train_tokens,
            "aofe": aofe,
            "aofe_ratio": aofe_ratio,
            **stats,
        }
        rows.append(row)
        write_csv(args.out_dir / "depth_scan_results.csv", rows)
        with (args.out_dir / "depth_scan_results.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(
            f"  AOFE={aofe:.6e} AOFE-ratio={aofe_ratio:.6f} "
            f"final_val_loss={stats['final_val_loss']:.4f}"
        )

    correlations = compute_correlations(rows)
    with (args.out_dir / "correlations.json").open("w", encoding="utf-8") as f:
        json.dump(correlations, f, indent=2)

    print("\nCorrelation summary:")
    for name, vals in correlations.items():
        print(f"  {name}: Pearson={vals['pearson']:.4f}, Spearman={vals['spearman']:.4f}")
    print(f"\nSaved results to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
