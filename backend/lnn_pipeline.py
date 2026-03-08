from __future__ import annotations

from contextlib import nullcontext
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import torch
from torch import nn


FEATURE_KEYS = ["target_pos", "target_vel", "actual_pos", "actual_vel", "controller_kp", "controller_kd", "tau_ff"]
TARGET_KEYS = ["actual_pos", "actual_vel", "torque"]


@dataclass
class RunConfig:
    hidden_units: int
    learning_rate: float
    window_size: int
    epochs: int
    batch_size: int = 1024
    pos_loss_weight: float = 1.0
    vel_loss_weight: float = 1.0
    torque_loss_weight: float = 0.35
    one_step_loss_weight: float = 1.0
    rollout_loss_weight: float = 0.0
    train_rollout_steps: int = 10
    controller_kp: float = 1.0
    controller_kd: float = 0.01
    precision: str = "bf16"
    compile_model: bool = False
    horizon_steps: int = 100
    device: str = "cpu"


def load_dataset(dataset_path: str) -> List[dict]:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or len(data) < 8:
        raise ValueError("Dataset must be a non-empty JSON array.")

    rows: List[dict] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "time": float(row.get("time", index * 0.01)),
                "target_pos": float(row.get("target_pos", 0.0) or 0.0),
                "target_vel": float(row.get("target_vel", 0.0) or 0.0),
                "actual_pos": float(row.get("actual_pos", 0.0) or 0.0),
                "actual_vel": float(row.get("actual_vel", 0.0) or 0.0),
                "torque": float(row.get("torque", 0.0) or 0.0),
                "tau_ff": float(row.get("tau_ff", 0.0) or 0.0),
            }
        )
    return rows


def dataset_to_tensors(dataset: List[dict], config: RunConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [
                row["target_pos"],
                row["target_vel"],
                row["actual_pos"],
                row["actual_vel"],
                config.controller_kp,
                config.controller_kd,
                row.get("tau_ff", 0.0),
            ]
            for row in dataset
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[row[key] for key in TARGET_KEYS] for row in dataset], dtype=torch.float32)
    return features, targets


def build_windows(
    features: torch.Tensor,
    targets: torch.Tensor,
    window_size: int,
    rollout_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    prev_targets = []
    rollout_targets = []
    rollout_features = []
    max_index = len(features) - rollout_steps + 1
    for idx in range(window_size, max_index):
        xs.append(features[idx - window_size : idx])
        ys.append(targets[idx])
        prev_targets.append(targets[idx - 1])
        rollout_targets.append(targets[idx : idx + rollout_steps])
        rollout_features.append(features[idx : idx + rollout_steps])
    return (
        torch.stack(xs),
        torch.stack(ys),
        torch.stack(prev_targets),
        torch.stack(rollout_targets),
        torch.stack(rollout_features),
    )


def split_windows(
    xs: torch.Tensor,
    ys: torch.Tensor,
    prev_targets: torch.Tensor,
    rollout_targets: torch.Tensor,
    rollout_features: torch.Tensor,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    total = len(xs)
    train_end = max(int(total * 0.7), 1)
    val_end = max(int(total * 0.85), train_end + 1)
    val_end = min(val_end, total - 1) if total > 2 else total

    return {
        "train": (xs[:train_end], ys[:train_end], prev_targets[:train_end], rollout_targets[:train_end], rollout_features[:train_end]),
        "val": (xs[train_end:val_end], ys[train_end:val_end], prev_targets[train_end:val_end], rollout_targets[train_end:val_end], rollout_features[train_end:val_end]),
        "test": (xs[val_end:], ys[val_end:], prev_targets[val_end:], rollout_targets[val_end:], rollout_features[val_end:]),
    }


def encode_deltas(prev_targets: torch.Tensor, next_targets: torch.Tensor) -> torch.Tensor:
    return next_targets - prev_targets


def decode_deltas(prev_targets: torch.Tensor, delta_targets: torch.Tensor) -> torch.Tensor:
    return prev_targets + delta_targets


def make_norm_stats(train_x: torch.Tensor, train_prev_targets: torch.Tensor, train_y: torch.Tensor) -> Dict[str, torch.Tensor]:
    x_mean = train_x.mean(dim=(0, 1))
    x_std = train_x.std(dim=(0, 1)).clamp_min(1e-6)
    train_delta_y = encode_deltas(train_prev_targets, train_y)
    y_mean = train_delta_y.mean(dim=0)
    y_std = train_delta_y.std(dim=0).clamp_min(1e-6)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def normalize_windows(xs: torch.Tensor, delta_targets: torch.Tensor, stats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    norm_x = (xs - stats["x_mean"]) / stats["x_std"]
    norm_y = (delta_targets - stats["y_mean"]) / stats["y_std"]
    return norm_x, norm_y


def normalize_targets(delta_targets: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    return (delta_targets - stats["y_mean"]) / stats["y_std"]


def compute_multitask_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    pos_loss_weight: float,
    vel_loss_weight: float,
    torque_loss_weight: float,
) -> torch.Tensor:
    pos_loss = torch.mean((pred_norm[:, 0] - target_norm[:, 0]) ** 2)
    vel_loss = torch.mean((pred_norm[:, 1] - target_norm[:, 1]) ** 2)
    torque_loss = torch.mean((pred_norm[:, 2] - target_norm[:, 2]) ** 2)
    return pos_loss_weight * pos_loss + vel_loss_weight * vel_loss + torque_loss_weight * torque_loss


def compute_rollout_loss(
    model: nn.Module,
    window_raw: torch.Tensor,
    prev_targets_raw: torch.Tensor,
    rollout_targets_raw: torch.Tensor,
    rollout_features_raw: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    pos_loss_weight: float,
    vel_loss_weight: float,
    torque_loss_weight: float,
) -> torch.Tensor:
    current_window_raw = window_raw.clone()
    current_prev_targets_raw = prev_targets_raw.clone()
    losses: List[torch.Tensor] = []

    for step in range(rollout_targets_raw.shape[1]):
        norm_window = (current_window_raw - stats["x_mean"]) / stats["x_std"]
        pred_norm = model(norm_window)
        target_delta_norm = normalize_targets(encode_deltas(current_prev_targets_raw, rollout_targets_raw[:, step]), stats)
        losses.append(
            compute_multitask_loss(
                pred_norm,
                target_delta_norm,
                pos_loss_weight,
                vel_loss_weight,
                torque_loss_weight,
            )
        )
        pred_delta_denorm = pred_norm * stats["y_std"] + stats["y_mean"]
        pred_abs = decode_deltas(current_prev_targets_raw, pred_delta_denorm)
        current_prev_targets_raw = pred_abs
        if step < rollout_targets_raw.shape[1] - 1:
            next_feature = rollout_features_raw[:, step].clone()
            next_feature[:, 2] = pred_abs[:, 0]
            next_feature[:, 3] = pred_abs[:, 1]
            current_window_raw = torch.cat([current_window_raw[:, 1:], next_feature.unsqueeze(1)], dim=1)

    return torch.stack(losses).mean()


class LiquidCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau_linear = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        tau = torch.sigmoid(self.tau_linear(torch.cat([x_t, h_prev], dim=-1)))
        proposal = torch.tanh(self.input_linear(x_t) + self.hidden_linear(h_prev))
        return h_prev + tau * (proposal - h_prev)


class LiquidNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.cell = LiquidCell(input_size, hidden_size)
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size = seq.shape[0]
        hidden = torch.zeros(batch_size, self.cell.hidden_size, device=seq.device)
        for step in range(seq.shape[1]):
            hidden = self.cell(seq[:, step], hidden)
        return self.readout(hidden)


def make_model(config: RunConfig) -> LiquidNet:
    return LiquidNet(len(FEATURE_KEYS), config.hidden_units, len(TARGET_KEYS))


def resolve_autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    xs_raw: torch.Tensor,
    ys_raw: torch.Tensor,
    prev_targets_raw: torch.Tensor,
    rollout_targets_raw: torch.Tensor,
    rollout_features_raw: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    batch_size: int,
    pos_loss_weight: float,
    vel_loss_weight: float,
    torque_loss_weight: float,
    one_step_loss_weight: float,
    rollout_loss_weight: float,
    precision: str,
    scaler: torch.amp.GradScaler | None,
) -> float:
    model.train()
    permutation = torch.randperm(len(xs_raw), device=xs_raw.device)
    losses: List[float] = []
    for start in range(0, len(xs_raw), batch_size):
        indices = permutation[start : start + batch_size]
        batch_x_raw = xs_raw[indices]
        batch_y_raw = ys_raw[indices]
        batch_prev_targets_raw = prev_targets_raw[indices]
        batch_rollout_targets_raw = rollout_targets_raw[indices]
        batch_rollout_features_raw = rollout_features_raw[indices]
        optimizer.zero_grad()
        with resolve_autocast_context(batch_x_raw.device, precision):
            batch_delta_y = encode_deltas(batch_prev_targets_raw, batch_y_raw)
            batch_x, batch_y = normalize_windows(batch_x_raw, batch_delta_y, stats)
            pred = model(batch_x)
            one_step_loss = compute_multitask_loss(
                pred,
                batch_y,
                pos_loss_weight,
                vel_loss_weight,
                torque_loss_weight,
            )
            rollout_loss = (
                compute_rollout_loss(
                    model,
                    batch_x_raw,
                    batch_prev_targets_raw,
                    batch_rollout_targets_raw,
                    batch_rollout_features_raw,
                    stats,
                    pos_loss_weight,
                    vel_loss_weight,
                    torque_loss_weight,
                )
                if rollout_loss_weight > 0
                else torch.zeros((), device=batch_x_raw.device, dtype=pred.dtype)
            )
            loss = one_step_loss_weight * one_step_loss + rollout_loss_weight * rollout_loss
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    xs_raw: torch.Tensor,
    ys_raw: torch.Tensor,
    prev_targets_raw: torch.Tensor,
    rollout_targets_raw: torch.Tensor,
    rollout_features_raw: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    pos_loss_weight: float,
    vel_loss_weight: float,
    torque_loss_weight: float,
    one_step_loss_weight: float,
    rollout_loss_weight: float,
    precision: str,
) -> float:
    if len(xs_raw) == 0:
        return 0.0
    model.eval()
    with resolve_autocast_context(xs_raw.device, precision):
        delta_ys = encode_deltas(prev_targets_raw, ys_raw)
        xs, ys = normalize_windows(xs_raw, delta_ys, stats)
        pred = model(xs)
        one_step_loss = compute_multitask_loss(
            pred,
            ys,
            pos_loss_weight,
            vel_loss_weight,
            torque_loss_weight,
        )
        rollout_loss = (
            compute_rollout_loss(
                model,
                xs_raw,
                prev_targets_raw,
                rollout_targets_raw,
                rollout_features_raw,
                stats,
                pos_loss_weight,
                vel_loss_weight,
                torque_loss_weight,
            )
            if rollout_loss_weight > 0
            else torch.zeros((), device=xs_raw.device, dtype=pred.dtype)
        )
        loss = one_step_loss_weight * one_step_loss + rollout_loss_weight * rollout_loss
    return float(loss.cpu())


@torch.no_grad()
def predict_single_step(model: nn.Module, xs: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    pred_norm = model(xs)
    return pred_norm * stats["y_std"] + stats["y_mean"]


@torch.no_grad()
def build_prediction_payload(
    model: nn.Module,
    dataset: List[dict],
    raw_features: torch.Tensor,
    raw_targets: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    config: RunConfig,
    eval_mode: Literal["one_step", "rollout"] = "rollout",
) -> Tuple[List[dict], Dict[str, float]]:
    model.eval()
    model_device = next(model.parameters()).device
    predictions: List[dict] = []
    for row in dataset:
        predictions.append({**row, "predicted_pos": None, "predicted_vel": None, "predicted_torque": None})

    windows_x, _, prev_targets, _, _ = build_windows(raw_features, raw_targets, config.window_size, max(1, config.train_rollout_steps))
    windows_x = windows_x.to(model_device)
    prev_targets = prev_targets.to(model_device)
    norm_x = (windows_x - stats["x_mean"]) / stats["x_std"]
    single_step_delta = predict_single_step(model, norm_x, stats)
    single_step = decode_deltas(prev_targets, single_step_delta).cpu()

    teacher_predictions = {}
    for idx, pred in enumerate(single_step, start=config.window_size):
        teacher_predictions[idx] = {
            "predicted_pos": float(pred[0]),
            "predicted_vel": float(pred[1]),
            "predicted_torque": float(pred[2]),
        }
        predictions[idx].update(teacher_predictions[idx])

    teacher_pos = []
    teacher_vel = []
    teacher_torque = []
    for idx in range(config.window_size, len(dataset)):
        pred = teacher_predictions.get(idx)
        if pred is None:
            continue
        teacher_pos.append((pred["predicted_pos"] - dataset[idx]["actual_pos"]) ** 2)
        teacher_vel.append((pred["predicted_vel"] - dataset[idx]["actual_vel"]) ** 2)
        teacher_torque.append((pred["predicted_torque"] - dataset[idx]["torque"]) ** 2)

    rollout_errors: List[float] = []
    rollout_vel_errors: List[float] = []
    rollout_torque_errors: List[float] = []
    if eval_mode == "rollout" and config.horizon_steps > 0:
        for idx in range(config.window_size, len(predictions)):
            predictions[idx]["predicted_pos"] = None
            predictions[idx]["predicted_vel"] = None
            predictions[idx]["predicted_torque"] = None
        segment_steps = max(1, config.horizon_steps)
        segment_starts = list(range(config.window_size, len(dataset), segment_steps))
        if segment_starts:
            for segment_start in segment_starts:
                predictions[segment_start]["predicted_pos"] = float(dataset[segment_start]["actual_pos"])
                predictions[segment_start]["predicted_vel"] = float(dataset[segment_start]["actual_vel"])
                predictions[segment_start]["predicted_torque"] = float(dataset[segment_start]["torque"])

            initial_windows = []
            prev_abs_targets = []
            segment_ends = []
            for segment_start in segment_starts:
                initial_windows.append(raw_features[segment_start - config.window_size + 1 : segment_start + 1])
                prev_abs_targets.append(raw_targets[segment_start])
                segment_ends.append(min(len(dataset), segment_start + segment_steps))

            current_windows = torch.stack(initial_windows).to(model_device)
            current_prev_targets = torch.stack(prev_abs_targets).to(model_device)
            segment_ends_tensor = torch.tensor(segment_ends, device=model_device)
            segment_starts_tensor = torch.tensor(segment_starts, device=model_device)

            for step_offset in range(1, segment_steps):
                target_indices = segment_starts_tensor + step_offset
                active_mask = target_indices < segment_ends_tensor
                if not bool(active_mask.any()):
                    break

                active_windows = current_windows[active_mask]
                active_prev_targets = current_prev_targets[active_mask]
                active_target_indices = target_indices[active_mask]
                active_target_indices_cpu = active_target_indices.cpu()

                norm_windows = (active_windows - stats["x_mean"]) / stats["x_std"]
                pred_delta = predict_single_step(model, norm_windows, stats)
                pred_abs = decode_deltas(active_prev_targets, pred_delta)
                pred_abs_cpu = pred_abs.cpu()

                for local_idx, global_idx_tensor in enumerate(active_target_indices_cpu):
                    global_idx = int(global_idx_tensor.item())
                    pred = pred_abs_cpu[local_idx]
                    predictions[global_idx]["predicted_pos"] = float(pred[0])
                    predictions[global_idx]["predicted_vel"] = float(pred[1])
                    predictions[global_idx]["predicted_torque"] = float(pred[2])
                    rollout_errors.append((float(pred[0]) - dataset[global_idx]["actual_pos"]) ** 2)
                    rollout_vel_errors.append((float(pred[1]) - dataset[global_idx]["actual_vel"]) ** 2)
                    rollout_torque_errors.append((float(pred[2]) - dataset[global_idx]["torque"]) ** 2)

                next_features = raw_features[active_target_indices_cpu].clone().to(model_device)
                next_features[:, 2] = pred_abs[:, 0]
                next_features[:, 3] = pred_abs[:, 1]
                current_windows[active_mask] = torch.cat([active_windows[:, 1:], next_features.unsqueeze(1)], dim=1)
                current_prev_targets[active_mask] = pred_abs

    metrics = {
        "evalMode": eval_mode,
        "posRmse": (sum(teacher_pos) / max(len(teacher_pos), 1)) ** 0.5,
        "velRmse": (sum(teacher_vel) / max(len(teacher_vel), 1)) ** 0.5,
        "torqueRmse": (sum(teacher_torque) / max(len(teacher_torque), 1)) ** 0.5,
        "rolloutRmse": (sum(rollout_errors) / max(len(rollout_errors), 1)) ** 0.5,
        "rolloutVelRmse": (sum(rollout_vel_errors) / max(len(rollout_vel_errors), 1)) ** 0.5,
        "rolloutTorqueRmse": (sum(rollout_torque_errors) / max(len(rollout_torque_errors), 1)) ** 0.5,
    }
    return predictions, metrics


def dump_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_checkpoint(run_dir: Path, model: nn.Module, stats: Dict[str, torch.Tensor], config: RunConfig) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "stats": {key: value.cpu() for key, value in stats.items()},
        "config": asdict(config),
    }
    torch.save(checkpoint, run_dir / "model.pt")


def load_checkpoint(run_dir: Path, device: str) -> Tuple[LiquidNet, Dict[str, torch.Tensor], RunConfig]:
    checkpoint = torch.load(run_dir / "model.pt", map_location=device)
    config = RunConfig(**checkpoint["config"])
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    stats = {key: value.to(device) for key, value in checkpoint["stats"].items()}
    return model, stats, config
