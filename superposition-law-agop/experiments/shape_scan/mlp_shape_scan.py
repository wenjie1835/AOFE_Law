"""
MLP teacher-student depth/width scan with AOFE metrics.

This script is designed to test whether MLPs show an analogous benign
superposition / optimal shape effect under a controlled feature-learning regime.

Task
----
A fixed hierarchical teacher MLP generates scalar regression targets:

    x ~ N(0, I)
    y = normalized_teacher(x) + noise

Student MLPs with different numbers of hidden layers are trained to imitate the
same teacher. For each student depth, the hidden width is chosen automatically
so that the parameter count is close to a fixed budget.

By default, every model sees:

    train_samples = tokens_per_param * parameter_count

with tokens_per_param=20, matching the Chinchilla-style N=20D regime you used
for the Transformer shape scan. Here N means synthetic training samples and D
means trainable student parameters. Default parameter budget is 3M; calibration,
validation, and AOFE evaluation sample defaults scale in the same proportion vs
the former 0.3M setup.

AOFE
----
For a scalar-output MLP, the input AGOP is cleanly:

    AGOP = E_x[ grad_x f(x) grad_x f(x)^T ]

AOFE and AOFE-ratio follow the same convention as data_scaling.py:

    AOFE       = ||AGOP - diag(AGOP)||_F^2
    AOFE-ratio = AOFE / ||AGOP||_F^2

Outputs
-------
The script writes:

    results_mlp_teacher_student_depth_scan/depth_scan_results.csv
    results_mlp_teacher_student_depth_scan/depth_scan_results.json
    results_mlp_teacher_student_depth_scan/correlations.json

Examples
--------
Inspect parameter-matched student configs:

    python3 mlp_teacher_student_depth_scan.py --dry-run-configs

Quick smoke test:

    python3 mlp_teacher_student_depth_scan.py --layers 2 4 --target-params 50000 --max-steps 20

Larger experiment (defaults are already ~3M params; override if needed):

    python3 mlp_teacher_student_depth_scan.py --layers 2 20
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
class MLPConfig:
    input_dim: int = 64
    output_dim: int = 1
    hidden_layers: int = 4
    hidden_width: int = 256
    activation: str = "gelu"
    bias: bool = True


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        if cfg.hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        self.cfg = cfg
        dims = [cfg.input_dim] + [cfg.hidden_width] * cfg.hidden_layers + [cfg.output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], bias=cfg.bias) for i in range(len(dims) - 1)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation == "gelu":
            return F.gelu(x)
        if self.cfg.activation == "relu":
            return F.relu(x)
        if self.cfg.activation == "tanh":
            return torch.tanh(x)
        raise ValueError(f"Unknown activation: {self.cfg.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self._act(layer(x))
        return self.layers[-1](x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TeacherTask:
    def __init__(
        self,
        teacher: MLP,
        *,
        input_dim: int,
        noise_std: float,
        device: torch.device,
        norm_mean: float,
        norm_std: float,
    ):
        self.teacher = teacher
        self.input_dim = input_dim
        self.noise_std = noise_std
        self.device = device
        self.norm_mean = norm_mean
        self.norm_std = max(norm_std, 1e-8)

    @torch.no_grad()
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(batch_size, self.input_dim, device=self.device)
        raw = self.teacher(x)
        y = (raw - self.norm_mean) / self.norm_std
        if self.noise_std > 0:
            y = y + self.noise_std * torch.randn_like(y)
        return x, y

    @torch.no_grad()
    def make_fixed_set(self, num_samples: int, batch_size: int = 8192) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        remaining = num_samples
        while remaining > 0:
            bsz = min(batch_size, remaining)
            x, y = self.sample(bsz)
            xs.append(x.cpu())
            ys.append(y.cpu())
            remaining -= bsz
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


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


def mlp_param_count(input_dim: int, output_dim: int, hidden_layers: int, width: int, bias: bool) -> int:
    dims = [input_dim] + [width] * hidden_layers + [output_dim]
    total = 0
    for din, dout in zip(dims[:-1], dims[1:]):
        total += din * dout
        if bias:
            total += dout
    return total


def choose_width_for_param_budget(
    *,
    input_dim: int,
    output_dim: int,
    hidden_layers: int,
    target_params: int,
    width_multiple: int,
    bias: bool,
    min_width: int,
    max_width: int,
) -> Tuple[int, int]:
    best_width = min_width
    best_params = mlp_param_count(input_dim, output_dim, hidden_layers, best_width, bias)
    best_err = abs(best_params - target_params)
    start = max(width_multiple, math.ceil(min_width / width_multiple) * width_multiple)
    for width in range(start, max_width + 1, width_multiple):
        params = mlp_param_count(input_dim, output_dim, hidden_layers, width, bias)
        err = abs(params - target_params)
        if err < best_err:
            best_width = width
            best_params = params
            best_err = err
    return best_width, best_params


def build_teacher(
    *,
    input_dim: int,
    teacher_depth: int,
    teacher_width: int,
    activation: str,
    seed: int,
    device: torch.device,
    noise_std: float,
    calibration_samples: int,
) -> TeacherTask:
    set_seed(seed)
    cfg = MLPConfig(
        input_dim=input_dim,
        output_dim=1,
        hidden_layers=teacher_depth,
        hidden_width=teacher_width,
        activation=activation,
        bias=True,
    )
    teacher = MLP(cfg).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Normalize teacher targets so loss scales are comparable across settings.
    vals = []
    remaining = calibration_samples
    with torch.no_grad():
        while remaining > 0:
            bsz = min(8192, remaining)
            x = torch.randn(bsz, input_dim, device=device)
            vals.append(teacher(x).detach())
            remaining -= bsz
        y = torch.cat(vals, dim=0)
        mean = float(y.mean().cpu())
        std = float(y.std(unbiased=False).cpu())

    return TeacherTask(
        teacher,
        input_dim=input_dim,
        noise_std=noise_std,
        device=device,
        norm_mean=mean,
        norm_std=std,
    )


@torch.no_grad()
def evaluate_mse(
    model: MLP,
    x_eval_cpu: torch.Tensor,
    y_eval_cpu: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    for start in range(0, x_eval_cpu.shape[0], batch_size):
        x = x_eval_cpu[start : start + batch_size].to(device)
        y = y_eval_cpu[start : start + batch_size].to(device)
        pred = model(x)
        loss_sum = F.mse_loss(pred, y, reduction="sum")
        total_loss += float(loss_sum.cpu())
        total += y.numel()
    model.train()
    return total_loss / max(1, total)


def configure_optimizer(
    model: MLP,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float] = (0.9, 0.95),
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
        betas=betas,
    )


def cosine_lr(step: int, max_steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def train_student(
    *,
    model: MLP,
    task: TeacherTask,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    train_samples: int,
    batch_size: int,
    grad_accum: int,
    eval_interval: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    warmup_steps: int,
    max_steps: Optional[int],
    device: torch.device,
) -> Dict[str, float]:
    model.to(device)
    opt = configure_optimizer(model, lr=lr, weight_decay=weight_decay)
    samples_per_step = batch_size * grad_accum
    planned_steps = math.ceil(train_samples / samples_per_step)
    steps = min(planned_steps, max_steps) if max_steps is not None else planned_steps
    best_val_loss = float("inf")
    last_train_loss = float("nan")
    start_time = time.time()
    print(
        f"  training steps={steps} planned_steps={planned_steps} "
        f"samples_per_step={samples_per_step}"
    )

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
        "samples_per_step": float(samples_per_step),
        "effective_train_samples": float(steps * samples_per_step),
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


def compute_input_aofe(
    *,
    model: MLP,
    x_eval_cpu: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    input_dim = model.cfg.input_dim
    agop = torch.zeros((input_dim, input_dim), device=device, dtype=torch.float32)
    total = 0
    for start in range(0, x_eval_cpu.shape[0], batch_size):
        x = x_eval_cpu[start : start + batch_size].to(device).detach().requires_grad_(True)
        with torch.enable_grad():
            out = model(x)
            grad = torch.autograd.grad(out.sum(), x, retain_graph=False, create_graph=False)[0]
        g = grad.to(torch.float32)
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
        "hidden_layers",
        "hidden_width",
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
    parser = argparse.ArgumentParser(description="MLP teacher-student depth/width scan with AOFE metrics.")
    parser.add_argument("--target-params", type=int, default=3_000_000)
    parser.add_argument("--samples-per-param", type=float, default=20.0)
    parser.add_argument("--train-samples", type=int, default=None, help="Override N=samples_per_param*params.")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 20], help="Either START END or explicit depths.")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--teacher-depth", type=int, default=4)
    parser.add_argument("--teacher-width", type=int, default=256)
    parser.add_argument("--activation", choices=["gelu", "relu", "tanh"], default="gelu")
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--calibration-samples", type=int, default=500_000)
    parser.add_argument("--val-samples", type=int, default=500_000)
    parser.add_argument("--aofe-samples", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--width-multiple", type=int, default=8)
    parser.add_argument("--min-width", type=int, default=8)
    parser.add_argument("--max-width", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--teacher-seed", type=int, default=2027)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("./results_mlp_teacher_student_depth_scan"))
    parser.add_argument("--max-steps", type=int, default=None, help="Debug cap; this breaks the intended N=20D budget.")
    parser.add_argument("--dry-run-configs", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    layers = parse_layers(args.layers)
    configs = []
    for depth in layers:
        width, params = choose_width_for_param_budget(
            input_dim=args.input_dim,
            output_dim=1,
            hidden_layers=depth,
            target_params=args.target_params,
            width_multiple=args.width_multiple,
            bias=True,
            min_width=args.min_width,
            max_width=args.max_width,
        )
        configs.append((depth, width, params))

    print("Parameter-matched student configs:")
    for depth, width, params in configs:
        rel = 100.0 * (params - args.target_params) / args.target_params
        print(f"  L={depth:2d} width={width:4d} params={params:,} ({rel:+.2f}%)")
    print(f"Device: {device}")
    if args.dry_run_configs:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        payload = vars(args).copy()
        payload["out_dir"] = str(args.out_dir)
        payload["device_resolved"] = str(device)
        json.dump(payload, f, indent=2)

    print("\nBuilding teacher task")
    task = build_teacher(
        input_dim=args.input_dim,
        teacher_depth=args.teacher_depth,
        teacher_width=args.teacher_width,
        activation=args.activation,
        seed=args.teacher_seed,
        device=device,
        noise_std=args.noise_std,
        calibration_samples=args.calibration_samples,
    )
    print(f"  teacher target mean={task.norm_mean:.6f} std={task.norm_std:.6f}")

    set_seed(args.seed + 17)
    print(f"Creating fixed validation set: {args.val_samples:,} samples")
    x_val, y_val = task.make_fixed_set(args.val_samples)
    set_seed(args.seed + 23)
    print(f"Creating fixed AOFE set: {args.aofe_samples:,} samples")
    x_aofe, _ = task.make_fixed_set(args.aofe_samples)

    rows: List[Dict[str, object]] = []
    for depth, width, params in configs:
        print("\n" + "=" * 80)
        print(f"student hidden_layers={depth} hidden_width={width} params={params:,}")
        set_seed(args.seed + depth)
        student_cfg = MLPConfig(
            input_dim=args.input_dim,
            output_dim=1,
            hidden_layers=depth,
            hidden_width=width,
            activation=args.activation,
            bias=True,
        )
        student = MLP(student_cfg)
        train_samples = args.train_samples
        if train_samples is None:
            train_samples = int(round(args.samples_per_param * params))
        stats = train_student(
            model=student,
            task=task,
            x_val=x_val,
            y_val=y_val,
            train_samples=train_samples,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            eval_interval=args.eval_interval,
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            device=device,
        )
        aofe, aofe_ratio, total_energy = compute_input_aofe(
            model=student,
            x_eval_cpu=x_aofe,
            batch_size=args.batch_size,
            device=device,
        )
        log10_aofe = math.log10(max(aofe, 1e-30))
        benign_score = aofe_ratio / max(log10_aofe, 1e-8)
        row: Dict[str, object] = {
            "hidden_layers": depth,
            "hidden_width": width,
            "depth_width_ratio": depth / width,
            "param_count": params,
            "param_error": params - args.target_params,
            "train_sample_budget": train_samples,
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
