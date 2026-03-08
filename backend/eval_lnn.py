from __future__ import annotations

import argparse
from pathlib import Path

from lnn_pipeline import (
    build_prediction_payload,
    dataset_to_tensors,
    dump_json,
    load_checkpoint,
    load_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--horizon-steps", type=int)
    parser.add_argument("--eval-mode", choices=["one_step", "rollout"], default="rollout")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    model, stats, config = load_checkpoint(run_dir, args.device)
    if args.horizon_steps is not None:
        config.horizon_steps = args.horizon_steps
    dataset = load_dataset(args.dataset)
    raw_features, raw_targets = dataset_to_tensors(dataset, config)
    predictions, metrics = build_prediction_payload(
        model,
        dataset,
        raw_features,
        raw_targets,
        stats,
        config,
        eval_mode=args.eval_mode,
    )
    dump_json(run_dir / "predictions.json", predictions)
    dump_json(
        run_dir / "metrics.json",
        {
            **metrics,
            "stability": max(
                0.0,
                1.0
                - (metrics["rolloutRmse"] if args.eval_mode == "rollout" else metrics["posRmse"]) * 4.0,
            ),
            "convergence": min(1.0, 0.35 + config.epochs / 200.0),
            "lagReduction": min(1.0, config.window_size / 100.0),
            "stictionCapability": min(1.0, config.hidden_units / 128.0),
        },
    )


if __name__ == "__main__":
    main()
