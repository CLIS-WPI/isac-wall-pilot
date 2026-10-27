"""CLI entry point.

Phase 0 implements ``preflight``. The current milestone implements
``validate`` for the Phase 2/3 checks, Phase 4 finite-wall diagnostics, and
Phase 5 continuous-waveform replay checks and Phase 6 measured receiver checks;
``run`` and ``report`` follow in later phases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _preflight(args: argparse.Namespace) -> int:
    from .environment import run_preflight

    return run_preflight(require_h100=args.require_h100, output_path=args.output)


def _validate(args: argparse.Namespace) -> int:
    from .experiments import run_validation

    return run_validation(
        config_path=args.config,
        require_h100=args.require_h100,
        output_path=args.output,
        phase4_output_path=args.phase4_output,
        phase5_output_path=args.phase5_output,
        phase6_output_path=args.phase6_output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isac_verify",
        description="GPU-only Sionna pilot for active wall-assisted sensing verification",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="check CUDA/OptiX, run PathSolver and RCSSolver, write environment.json",
    )
    preflight.add_argument(
        "--require-h100",
        action="store_true",
        help="fail unless the visible GPU name contains H100",
    )
    preflight.add_argument(
        "--output",
        default="results/preflight",
        help="directory for environment.json and preflight.log",
    )
    preflight.set_defaults(handler=_preflight)

    validate = subparsers.add_parser(
        "validate",
        help="run Phase 2/3 validation and Phase 4/5/6 checks",
    )
    validate.add_argument(
        "--config",
        default="configs/pilot.yaml",
        help="pilot YAML configuration",
    )
    validate.add_argument(
        "--require-h100",
        action="store_true",
        help="fail unless the visible GPU name contains H100",
    )
    validate.add_argument(
        "--output",
        default="results/validate",
        help="directory for validation_summary.json and Experiment A figure",
    )
    validate.add_argument(
        "--phase4-output",
        default="results/phase4",
        help="directory for Phase 4 path diagnostics and geometry reference",
    )
    validate.add_argument(
        "--phase5-output",
        default="results/phase5",
        help="directory for Phase 5 continuous-waveform replay summary",
    )
    validate.add_argument(
        "--phase6-output",
        default="results/phase6",
        help="directory for Phase 6 measured receiver summary",
    )
    validate.set_defaults(handler=_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return int(handler(args))
    except Exception as exc:  # noqa: BLE001
        print(f"isac_verify: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
