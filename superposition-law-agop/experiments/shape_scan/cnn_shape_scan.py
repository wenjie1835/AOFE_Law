"""
CNN denoising autoencoder depth/width scan with AOFE metrics.

Purpose
-------
This experiment is the CNN analogue of the Transformer and MLP shape scans:
fix a parameter budget, vary depth, choose width automatically, and measure
whether reconstruction loss is related to AOFE / AOFE-ratio.

Task
----
The model is a residual CNN denoising autoencoder:

    corrupted procedural image -> CNN -> clean procedural image

Images are generated indefinitely, so the training regime can be controlled by
visual prediction positions rather than by the finite size of a dataset.

By default:

    train_visual_tokens = 20 * parameter_count
    visual_tokens_per_image = H * W
    train_images = ceil(train_visual_tokens / (H * W))

This is deliberately not "20 * images"; it treats each supervised spatial
prediction location as the visual-token unit. Use --count-channels-as-tokens if
you want D = images * H * W * C instead.

AOFE
----
For the image-to-image autoencoder, the output is high-dimensional. We estimate
the input AGOP using Hutchinson probes:

    J = d model(x) / d x
    AGOP_input = E_x,r [ (J^T r) (J^T r)^T ],  r ~ N(0, I)

Then:

    AOFE       = ||AGOP - diag(AGOP)||_F^2
    AOFE-ratio = AOFE / ||AGOP||_F^2

For 32x32 RGB images the input dimension is 3072, so the full AGOP is practical
on CPU/GPU memory. If you increase image size, reduce --aofe-samples/probes or
use a smaller --image-size.

Outputs
-------
    results_cnn_autoencoder_depth_scan/depth_scan_results.csv
    results_cnn_autoencoder_depth_scan/depth_scan_results.json
    results_cnn_autoencoder_depth_scan/correlations.json

Examples
--------
Inspect configs:

    python3 cnn_autoencoder_depth_scan.py --dry-run-configs

Quick smoke test:

    python3 cnn_autoencoder_depth_scan.py --layers 2 4 --target-params 50000 --max-steps 20 --aofe-samples 128

Full-ish scan:

    python3 cnn_autoencoder_depth_scan.py --target-params 3000000 --layers 2 24
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CNNConfig:
    image_size: int = 32
    in_channels: int = 3
    width: int = 192
    blocks: int = 8
    activation: str = "gelu"
    norm: str = "group"
    bias: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
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


def group_count(width: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, width), 0, -1):
        if width % g == 0:
            return g
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, width: int, *, activation: str, norm: str, bias: bool):
        super().__init__()
        self.activation = activation
        if norm == "group":
            self.norm1 = nn.GroupNorm(group_count(width), width)
            self.norm2 = nn.GroupNorm(group_count(width), width)
        elif norm == "batch":
            self.norm1 = nn.BatchNorm2d(width)
            self.norm2 = nn.BatchNorm2d(width)
        elif norm == "none":
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise ValueError(f"Unknown norm: {norm}")
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=bias)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self._act(self.norm1(x)))
        y = self.conv2(self._act(self.norm2(y)))
        return x + y / math.sqrt(2.0)


class ResidualAutoencoderCNN(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Conv2d(cfg.in_channels, cfg.width, kernel_size=3, padding=1, bias=cfg.bias)
        self.blocks = nn.ModuleList(
            ResidualBlock(cfg.width, activation=cfg.activation, norm=cfg.norm, bias=cfg.bias)
            for _ in range(cfg.blocks)
        )
        if cfg.norm == "group":
            self.out_norm = nn.GroupNorm(group_count(cfg.width), cfg.width)
        elif cfg.norm == "batch":
            self.out_norm = nn.BatchNorm2d(cfg.width)
        elif cfg.norm == "none":
            self.out_norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm: {cfg.norm}")
        self.out = nn.Conv2d(cfg.width, cfg.in_channels, kernel_size=3, padding=1, bias=cfg.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation == "gelu":
            return F.gelu(x)
        if self.cfg.activation == "relu":
            return F.relu(x)
        if self.cfg.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unknown activation: {self.cfg.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        h = self._act(self.out_norm(h))
        return self.out(h)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ProceduralDenoisingTask:
    def __init__(
        self,
        *,
        image_size: int,
        channels: int,
        shapes_per_image: int,
        noise_std: float,
        mask_prob: float,
        device: torch.device,
    ):
        if channels != 3:
            raise ValueError("The procedural generator currently expects RGB images (channels=3).")
        self.image_size = image_size
        self.channels = channels
        self.shapes_per_image = shapes_per_image
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.device = device
        coords = torch.linspace(-1.0, 1.0, image_size, device=device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.xx = xx.view(1, 1, image_size, image_size)
        self.yy = yy.view(1, 1, image_size, image_size)

    def sample_clean(self, batch_size: int) -> torch.Tensor:
        bsz = batch_size
        h = w = self.image_size
        device = self.device
        base = torch.rand(bsz, 3, 1, 1, device=device) * 0.25
        gx = torch.randn(bsz, 3, 1, 1, device=device) * 0.08
        gy = torch.randn(bsz, 3, 1, 1, device=device) * 0.08
        img = base + gx * self.xx + gy * self.yy

        # Smooth colored Gaussian blobs. They give the CNN local and mid-range
        # structure without relying on a finite image dataset.
        for _ in range(self.shapes_per_image):
            cx = torch.empty(bsz, 1, 1, 1, device=device).uniform_(-0.85, 0.85)
            cy = torch.empty(bsz, 1, 1, 1, device=device).uniform_(-0.85, 0.85)
            sx = torch.empty(bsz, 1, 1, 1, device=device).uniform_(0.06, 0.35)
            sy = torch.empty(bsz, 1, 1, 1, device=device).uniform_(0.06, 0.35)
            color = torch.rand(bsz, 3, 1, 1, device=device)
            amp = torch.empty(bsz, 1, 1, 1, device=device).uniform_(0.25, 0.9)
            blob = torch.exp(-0.5 * (((self.xx - cx) / sx) ** 2 + ((self.yy - cy) / sy) ** 2))
            img = img + amp * color * blob

        # Add a weak checker/stripe component, random per image, to create
        # high-frequency structure that cannot be solved by only smoothing.
        freq = torch.randint(2, 8, (bsz, 1, 1, 1), device=device, dtype=torch.float32)
        phase = torch.rand(bsz, 1, 1, 1, device=device) * (2 * math.pi)
        stripe = 0.04 * torch.sin(freq * math.pi * (self.xx + self.yy) + phase)
        img = img + stripe
        return img.clamp(0.0, 1.0)

    def corrupt(self, clean: torch.Tensor) -> torch.Tensor:
        noisy = clean + self.noise_std * torch.randn_like(clean)
        if self.mask_prob > 0:
            mask = (torch.rand(clean.shape[0], 1, clean.shape[2], clean.shape[3], device=clean.device) > self.mask_prob).to(clean.dtype)
            noisy = noisy * mask
        return noisy.clamp(0.0, 1.0)

    @torch.no_grad()
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.sample_clean(batch_size)
        corrupted = self.corrupt(clean)
        return corrupted, clean

    @torch.no_grad()
    def make_fixed_set(self, num_images: int, batch_size: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        remaining = num_images
        while remaining > 0:
            bsz = min(batch_size, remaining)
            x, y = self.sample(bsz)
            xs.append(x.cpu())
            ys.append(y.cpu())
            remaining -= bsz
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def count_params_for_config(cfg: CNNConfig) -> int:
    model = ResidualAutoencoderCNN(cfg)
    return model.parameter_count()


def choose_width_for_param_budget(
    *,
    blocks: int,
    target_params: int,
    image_size: int,
    channels: int,
    activation: str,
    norm: str,
    bias: bool,
    width_multiple: int,
    min_width: int,
    max_width: int,
) -> Tuple[CNNConfig, int]:
    best_cfg = None
    best_params = None
    best_err = None
    start = max(width_multiple, math.ceil(min_width / width_multiple) * width_multiple)
    for width in range(start, max_width + 1, width_multiple):
        cfg = CNNConfig(
            image_size=image_size,
            in_channels=channels,
            width=width,
            blocks=blocks,
            activation=activation,
            norm=norm,
            bias=bias,
        )
        params = count_params_for_config(cfg)
        err = abs(params - target_params)
        if best_err is None or err < best_err:
            best_cfg = cfg
            best_params = params
            best_err = err
    if best_cfg is None or best_params is None:
        raise RuntimeError("No valid width found for parameter budget.")
    return best_cfg, best_params


@torch.no_grad()
def evaluate_mse(
    model: ResidualAutoencoderCNN,
    x_eval_cpu: torch.Tensor,
    y_eval_cpu: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, x_eval_cpu.shape[0], batch_size):
        x = x_eval_cpu[start : start + batch_size].to(device)
        y = y_eval_cpu[start : start + batch_size].to(device)
        pred = model(x)
        total += float(F.mse_loss(pred, y, reduction="sum").cpu())
        count += y.numel()
    model.train()
    return total / max(1, count)


def configure_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def cosine_lr(step: int, max_steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def train_model(
    *,
    model: ResidualAutoencoderCNN,
    task: ProceduralDenoisingTask,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    train_images: int,
    batch_size: int,
    grad_accum: int,
    eval_interval: int,
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

    opt = configure_optimizer(model, lr=lr, weight_decay=weight_decay)
    images_per_step = batch_size * grad_accum
    planned_steps = math.ceil(train_images / images_per_step)
    steps = min(planned_steps, max_steps) if max_steps is not None else planned_steps
    best_val_loss = float("inf")
    last_train_loss = float("nan")
    start_time = time.time()
    print(f"  training steps={steps} planned_steps={planned_steps} images_per_step={images_per_step}")

    for step in range(steps):
        lr_now = cosine_lr(step, steps, lr, min_lr, warmup_steps)
        for group in opt.param_groups:
            group["lr"] = lr_now
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum):
            x, y = task.sample(batch_size)
            pred = model(x)
            loss = F.mse_loss(pred, y, reduction="mean") / grad_accum
            loss.backward()
            accum_loss += float(loss.detach().cpu()) * grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_train_loss = accum_loss

        if step == 0 or (step + 1) % eval_interval == 0 or step == steps - 1:
            val_loss = evaluate_mse(model, x_val, y_val, batch_size=batch_size, device=device)
            best_val_loss = min(best_val_loss, val_loss)
            elapsed = (time.time() - start_time) / 60.0
            print(
                f"    step={step + 1:6d}/{steps} "
                f"train_mse={last_train_loss:.6f} val_mse={val_loss:.6f} "
                f"lr={lr_now:.2e} elapsed={elapsed:.1f}m"
            )

    final_val_loss = evaluate_mse(model, x_val, y_val, batch_size=batch_size, device=device)
    best_val_loss = min(best_val_loss, final_val_loss)
    return {
        "steps": float(steps),
        "planned_steps": float(planned_steps),
        "images_per_step": float(images_per_step),
        "effective_train_images": float(steps * images_per_step),
        "final_train_loss": float(last_train_loss),
        "final_val_loss": float(final_val_loss),
        "best_val_loss": float(best_val_loss),
    }


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


def compute_hutchinson_input_aofe(
    *,
    model: ResidualAutoencoderCNN,
    x_eval_cpu: torch.Tensor,
    batch_size: int,
    probes: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    input_dim = model.cfg.in_channels * model.cfg.image_size * model.cfg.image_size
    agop = torch.zeros((input_dim, input_dim), device=device, dtype=torch.float32)
    total = 0

    for start in range(0, x_eval_cpu.shape[0], batch_size):
        base = x_eval_cpu[start : start + batch_size].to(device)
        for _ in range(probes):
            x = base.detach().requires_grad_(True)
            with torch.enable_grad():
                out = model(x)
                probe = torch.randn_like(out)
                scalar = (out * probe).sum()
                grad = torch.autograd.grad(scalar, x, retain_graph=False, create_graph=False)[0]
            g = grad.reshape(grad.shape[0], -1).to(torch.float32)
            agop += g.T @ g
            total += g.shape[0]

    agop /= max(1, total)
    agop = 0.5 * (agop + agop.T)
    aofe = offdiag_energy_from_matrix(agop)
    ratio = offdiag_energy_ratio_from_matrix(agop)
    total_energy = float((agop ** 2).sum().item())
    model.train()
    return aofe, ratio, total_energy


def pearsonr(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = [float(v) for v in xs]
    y = [float(v) for v in ys]
    if len(x) != len(y) or not x:
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    den = math.sqrt(vx * vy)
    if den == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / den


def rankdata(xs: Iterable[float]) -> List[float]:
    vals = [float(v) for v in xs]
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = 0.5 * (i + j)
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearmanr(xs: Iterable[float], ys: Iterable[float]) -> float:
    return pearsonr(rankdata(xs), rankdata(ys))


def compute_correlations(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    loss = [float(r["final_val_loss"]) for r in rows]
    out: Dict[str, Dict[str, float]] = {}
    for key in [
        "blocks",
        "width",
        "depth_width_ratio",
        "aofe",
        "log10_aofe",
        "aofe_ratio",
        "aofe_total_energy",
        "benign_score",
    ]:
        vals = [float(r[key]) for r in rows]
        out[f"final_val_loss_vs_{key}"] = {
            "pearson": pearsonr(loss, vals),
            "spearman": spearmanr(loss, vals),
        }
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_layers(values: List[int]) -> List[int]:
    if len(values) == 2:
        lo, hi = values
        if lo > hi:
            raise ValueError("--layers START END requires START <= END")
        return list(range(lo, hi + 1))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN denoising autoencoder depth/width scan with AOFE metrics.")
    parser.add_argument("--target-params", type=int, default=3_000_000)
    parser.add_argument("--visual-tokens-per-param", type=float, default=20.0)
    parser.add_argument("--train-images", type=int, default=None, help="Override image count; otherwise derived from visual-token budget.")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 24], help="Either START END or explicit block counts.")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--count-channels-as-tokens", action="store_true")
    parser.add_argument("--shapes-per-image", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=0.18)
    parser.add_argument("--mask-prob", type=float, default=0.12)
    parser.add_argument("--activation", choices=["gelu", "relu", "silu"], default="gelu")
    parser.add_argument("--norm", choices=["group", "batch", "none"], default="group")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--val-images", type=int, default=4096)
    parser.add_argument("--aofe-samples", type=int, default=512)
    parser.add_argument("--aofe-batch-size", type=int, default=16)
    parser.add_argument("--aofe-probes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--width-multiple", type=int, default=8)
    parser.add_argument("--min-width", type=int, default=16)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("./results_cnn_autoencoder_depth_scan"))
    parser.add_argument("--max-steps", type=int, default=None, help="Debug cap; this breaks the intended visual-token budget.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dry-run-configs", action="store_true")
    args = parser.parse_args()

    if args.image_size * args.image_size * args.channels > 4096:
        print(
            "Warning: full input AGOP is large for this image size. "
            "Consider reducing --aofe-samples, --aofe-probes, or --image-size."
        )

    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    blocks_list = parse_layers(args.layers)
    configs = []
    for blocks in blocks_list:
        cfg, params = choose_width_for_param_budget(
            blocks=blocks,
            target_params=args.target_params,
            image_size=args.image_size,
            channels=args.channels,
            activation=args.activation,
            norm=args.norm,
            bias=True,
            width_multiple=args.width_multiple,
            min_width=args.min_width,
            max_width=args.max_width,
        )
        configs.append((cfg, params))

    print("Parameter-matched CNN autoencoder configs:")
    for cfg, params in configs:
        rel = 100.0 * (params - args.target_params) / args.target_params
        print(f"  blocks={cfg.blocks:2d} width={cfg.width:4d} params={params:,} ({rel:+.2f}%)")
    print(f"Device: {device}")
    if args.dry_run_configs:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        payload = vars(args).copy()
        payload["out_dir"] = str(args.out_dir)
        payload["device_resolved"] = str(device)
        json.dump(payload, f, indent=2)

    set_seed(args.seed)
    task = ProceduralDenoisingTask(
        image_size=args.image_size,
        channels=args.channels,
        shapes_per_image=args.shapes_per_image,
        noise_std=args.noise_std,
        mask_prob=args.mask_prob,
        device=device,
    )

    set_seed(args.seed + 17)
    print(f"Creating fixed validation set: {args.val_images:,} images")
    x_val, y_val = task.make_fixed_set(args.val_images)
    set_seed(args.seed + 23)
    print(f"Creating fixed AOFE set: {args.aofe_samples:,} images")
    x_aofe, _ = task.make_fixed_set(args.aofe_samples)

    rows: List[Dict[str, object]] = []
    spatial_tokens_per_image = args.image_size * args.image_size
    if args.count_channels_as_tokens:
        spatial_tokens_per_image *= args.channels

    for cfg, params in configs:
        print("\n" + "=" * 80)
        print(f"student blocks={cfg.blocks} width={cfg.width} params={params:,}")
        set_seed(args.seed + cfg.blocks)
        model = ResidualAutoencoderCNN(cfg)

        train_visual_tokens = int(round(args.visual_tokens_per_param * params))
        if args.train_images is None:
            train_images = math.ceil(train_visual_tokens / spatial_tokens_per_image)
        else:
            train_images = args.train_images
            train_visual_tokens = train_images * spatial_tokens_per_image

        stats = train_model(
            model=model,
            task=task,
            x_val=x_val,
            y_val=y_val,
            train_images=train_images,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            eval_interval=args.eval_interval,
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            device=device,
            compile_model=args.compile,
        )

        aofe, aofe_ratio, total_energy = compute_hutchinson_input_aofe(
            model=model,
            x_eval_cpu=x_aofe,
            batch_size=args.aofe_batch_size,
            probes=args.aofe_probes,
            device=device,
        )
        log10_aofe = math.log10(max(aofe, 1e-30))
        benign_score = aofe_ratio / max(log10_aofe, 1e-8)

        row: Dict[str, object] = {
            "blocks": cfg.blocks,
            "width": cfg.width,
            "depth_width_ratio": cfg.blocks / cfg.width,
            "param_count": params,
            "param_error": params - args.target_params,
            "train_visual_token_budget": train_visual_tokens,
            "visual_tokens_per_image": spatial_tokens_per_image,
            "train_image_budget": train_images,
            "aofe": aofe,
            "log10_aofe": log10_aofe,
            "aofe_ratio": aofe_ratio,
            "aofe_total_energy": total_energy,
            "benign_score": benign_score,
            **stats,
        }
        rows.append(row)
        write_csv(args.out_dir / "depth_scan_results.csv", rows)
        with (args.out_dir / "depth_scan_results.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(
            f"  AOFE={aofe:.6e} log10_AOFE={log10_aofe:.3f} "
            f"AOFE-ratio={aofe_ratio:.6f} final_val_loss={stats['final_val_loss']:.6f}"
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
