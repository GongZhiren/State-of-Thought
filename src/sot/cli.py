"""Command-line interface for State-of-Thought."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import checkpoint_summary
from .data_validation import validate_data_config
from .evaluation import evaluate_config
from .judge import evaluate_judge_files, fit_judge_files
from .training import train_from_config


def _checkpoint_verify(args: argparse.Namespace) -> int:
    print(json.dumps(checkpoint_summary(args.path), indent=2, ensure_ascii=False))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    print(json.dumps(evaluate_config(args.config), indent=2, ensure_ascii=False))
    return 0


def _train(args: argparse.Namespace) -> int:
    print(train_from_config(args.config))
    return 0


def _data_validate(args: argparse.Namespace) -> int:
    report = validate_data_config(
        args.config,
        manifest_path=args.manifest,
        check_images=not args.skip_image_check,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 2


def _judge_evaluate(args: argparse.Namespace) -> int:
    summary = evaluate_judge_files(
        args.checkpoint, args.records, args.output, batch_size=args.batch_size
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _judge_train(args: argparse.Namespace) -> int:
    summary = fit_judge_files(
        args.records,
        args.output,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        pca_dim=args.pca_dim,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sot", description="State-of-Thought tools")
    commands = parser.add_subparsers(dest="command", required=True)
    checkpoint = commands.add_parser("checkpoint", help="inspect controller checkpoints")
    checkpoint_commands = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    verify = checkpoint_commands.add_parser("verify", help="validate a .pt controller")
    verify.add_argument("path", type=Path)
    verify.set_defaults(handler=_checkpoint_verify)
    evaluate = commands.add_parser("evaluate", help="run a resumable paper-aligned evaluation")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.set_defaults(handler=_evaluate)
    train = commands.add_parser("train", help="fit the controller from offline trajectories")
    train.add_argument("--config", type=Path, required=True)
    train.set_defaults(handler=_train)
    data = commands.add_parser("data", help="validate paper evaluation data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_validate = data_commands.add_parser(
        "validate", help="verify sample order, references, and image assets"
    )
    data_validate.add_argument("--config", type=Path, required=True)
    data_validate.add_argument("--manifest", type=Path)
    data_validate.add_argument(
        "--skip-image-check",
        action="store_true",
        help="validate VLM IDs and references without requiring local image files",
    )
    data_validate.set_defaults(handler=_data_validate)
    judge = commands.add_parser("judge", help="train or evaluate a trajectory-quality judge")
    judge_commands = judge.add_subparsers(dest="judge_command", required=True)
    judge_evaluate = judge_commands.add_parser("evaluate", help="evaluate a released judge")
    judge_evaluate.add_argument("--checkpoint", type=Path, required=True)
    judge_evaluate.add_argument("--records", type=Path, required=True)
    judge_evaluate.add_argument("--output", type=Path, required=True)
    judge_evaluate.add_argument("--batch-size", type=int, default=64)
    judge_evaluate.set_defaults(handler=_judge_evaluate)
    judge_train = judge_commands.add_parser("train", help="fit the PCA-trajectory MLP judge")
    judge_train.add_argument("--records", type=Path, required=True)
    judge_train.add_argument("--output", type=Path, required=True)
    judge_train.add_argument("--embedding-model", default="text-embedding-3-large")
    judge_train.add_argument("--batch-size", type=int, default=64)
    judge_train.add_argument("--pca-dim", type=int, default=32)
    judge_train.add_argument("--seed", type=int, default=42)
    judge_train.set_defaults(handler=_judge_train)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
