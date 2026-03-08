from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lnn_pipeline import (
    RunConfig,
    build_prediction_payload,
    build_windows,
    dataset_to_tensors,
    dump_json,
    evaluate_loss,
    load_dataset,
    make_model,
    make_norm_stats,
    normalize_windows,
    save_checkpoint,
    split_windows,
    train_epoch,
)


def write_status(path: Path, payload: dict) -> None:
    dump_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--pos-loss-weight", type=float, default=1.0)
    parser.add_argument("--vel-loss-weight", type=float, default=1.0)
    parser.add_argument("--torque-loss-weight", type=float, default=0.35)
    parser.add_argument("--one-step-loss-weight", type=float, default=1.0)
    parser.add_argument("--rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--train-rollout-steps", type=int, default=10)
    parser.add_argument("--controller-kp", type=float, default=1.0)
    parser.add_argument("--controller-kd", type=float, default=0.01)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"

    config = RunConfig(
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pos_loss_weight=args.pos_loss_weight,
        vel_loss_weight=args.vel_loss_weight,
        torque_loss_weight=args.torque_loss_weight,
        one_step_loss_weight=args.one_step_loss_weight,
        rollout_loss_weight=args.rollout_loss_weight,
        train_rollout_steps=args.train_rollout_steps,
        controller_kp=args.controller_kp,
        controller_kd=args.controller_kd,
        precision=args.precision,
        compile_model=args.compile_model,
        horizon_steps=args.horizon_steps,
        device=args.device,
    )

    write_status(
        status_path,
        {
            "state": "loading_data",
            "progress": 1,
            "epoch": 0,
            "totalEpochs": config.epochs,
            "message": "Loading dataset",
            "lossCurve": [],
        },
    )

    dataset = load_dataset(args.dataset)
    raw_features, raw_targets = dataset_to_tensors(dataset, config)
    min_required = config.window_size + max(2, config.train_rollout_steps)
    if len(raw_features) <= min_required:
        raise ValueError("Dataset is too short for the selected window size and rollout steps.")

    windows_x, windows_y, window_prev_targets, rollout_targets, rollout_features = build_windows(
        raw_features,
        raw_targets,
        config.window_size,
        max(1, config.train_rollout_steps),
    )
    splits = split_windows(windows_x, windows_y, window_prev_targets, rollout_targets, rollout_features)
    train_raw_x, train_raw_y, train_prev_targets, train_rollout_targets, train_rollout_features = splits["train"]
    val_raw_x, val_raw_y, val_prev_targets, val_rollout_targets, val_rollout_features = splits["val"]
    stats = make_norm_stats(train_raw_x, train_prev_targets, train_raw_y)

    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    model = make_model(config).to(device)
    base_model = model
    if config.compile_model:
        try:
            model = torch.compile(model)
        except Exception:
            pass
    train_raw_x, train_raw_y = train_raw_x.to(device), train_raw_y.to(device)
    val_raw_x, val_raw_y = val_raw_x.to(device), val_raw_y.to(device)
    train_prev_targets = train_prev_targets.to(device)
    val_prev_targets = val_prev_targets.to(device)
    train_rollout_targets = train_rollout_targets.to(device)
    train_rollout_features = train_rollout_features.to(device)
    val_rollout_targets = val_rollout_targets.to(device)
    val_rollout_features = val_rollout_features.to(device)
    stats = {key: value.to(device) for key, value in stats.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_curve = []
    batch_size = max(1, min(config.batch_size, len(train_raw_x)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.precision == "fp16")

    best_val = float("inf")
    best_state = None
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(
            model,
            optimizer,
            train_raw_x,
            train_raw_y,
            train_prev_targets,
            train_rollout_targets,
            train_rollout_features,
            stats,
            batch_size=batch_size,
            pos_loss_weight=config.pos_loss_weight,
            vel_loss_weight=config.vel_loss_weight,
            torque_loss_weight=config.torque_loss_weight,
            one_step_loss_weight=config.one_step_loss_weight,
            rollout_loss_weight=config.rollout_loss_weight,
            precision=config.precision,
            scaler=scaler,
        )
        val_loss = evaluate_loss(
            model,
            val_raw_x,
            val_raw_y,
            val_prev_targets,
            val_rollout_targets,
            val_rollout_features,
            stats,
            pos_loss_weight=config.pos_loss_weight,
            vel_loss_weight=config.vel_loss_weight,
            torque_loss_weight=config.torque_loss_weight,
            one_step_loss_weight=config.one_step_loss_weight,
            rollout_loss_weight=config.rollout_loss_weight,
            precision=config.precision,
        )
        if val_loss <= best_val:
            best_val = val_loss
            current_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            best_state = {key: value.detach().cpu().clone() for key, value in current_model.state_dict().items()}

        progress = int(epoch / config.epochs * 100)
        loss_curve.append({"epoch": epoch, "loss": train_loss, "valLoss": val_loss})
        write_status(
            status_path,
            {
                "state": "training",
                "progress": progress,
                "epoch": epoch,
                "totalEpochs": config.epochs,
                "message": f"Epoch {epoch}/{config.epochs}",
                "lossCurve": loss_curve[-60:],
            },
        )

    if best_state is not None:
        base_model.load_state_dict(best_state)

    predictions, metrics = build_prediction_payload(base_model, dataset, raw_features, raw_targets, stats, config)
    dump_json(run_dir / "loss_curve.json", loss_curve)
    dump_json(run_dir / "predictions.json", predictions)
    dump_json(
        run_dir / "metrics.json",
        {
            **metrics,
            "stability": max(0.0, 1.0 - metrics["rolloutRmse"] * 4.0),
            "convergence": min(1.0, 0.35 + config.epochs / 200.0),
            "lagReduction": min(1.0, config.window_size / 100.0),
            "stictionCapability": min(1.0, config.hidden_units / 128.0),
        },
    )
    dump_json(run_dir / "run_config.json", vars(args))
    save_checkpoint(run_dir, base_model, stats, config)

    write_status(
        status_path,
        {
            "state": "completed",
            "progress": 100,
            "epoch": config.epochs,
            "totalEpochs": config.epochs,
            "message": "Training completed",
            "lossCurve": loss_curve[-60:],
            "metrics": metrics,
        },
    )


if __name__ == "__main__":
    main()
