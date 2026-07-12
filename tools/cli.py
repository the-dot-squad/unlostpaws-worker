"""
Unified operator CLI for the vision worker.

Subcommands:
  doctor    — detect hardware and recommend VISION_PROFILE
  smoke     — end-to-end pipeline test for one profile
  benchmark — per-stage latency (one profile or batch JSON)
  export    — ONNX export for maintainers (CI)
  validate  — ONNX vs PyTorch accuracy check (CI)
"""

from __future__ import annotations

import argparse
import asyncio
import json

from tools._console import err, out
from tools.benchmark import DEFAULT_PROFILES, run_benchmarks, run_single
from tools.doctor import detect_and_recommend, print_report, validate_profile
from tools.eval import run_eval
from tools.export import main as export_main
from tools.smoke import run_smoke
from tools.validate import main as validate_main
from tools._paths import ROOT


def _cmd_doctor(args: argparse.Namespace) -> int:
    if args.profile and not args.smoke:
        validate_profile(args.profile)
        return 0

    if args.profile and args.smoke:
        validate_profile(args.profile)
        result = asyncio.run(run_smoke(args.profile))
        out(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1

    hardware, config = detect_and_recommend()
    print_report(hardware, config)
    try:
        validate_profile(config.vision_profile)
    except SystemExit as exc:
        # Recommendation may not be runnable on this host (e.g. GPU profile on a laptop).
        err(
            f"\nNote: preflight failed for VISION_PROFILE={config.vision_profile} on this host."
        )
        code = exc.code if isinstance(exc.code, int) else 1
        return code
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    validate_profile(args.profile)
    result = asyncio.run(run_smoke(args.profile))
    out(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    images = ROOT / args.images
    if args.all:
        profiles = args.profiles or DEFAULT_PROFILES
        output = ROOT / args.output if args.output else None
        return asyncio.run(run_benchmarks(profiles, images, args.runs, output))
    return run_single(args.profile, images, args.runs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools",
        description="UnLostPaws vision worker operator tools",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Detect hardware and recommend a profile")
    doctor.add_argument("--profile", help="Validate a specific profile (preflight)")
    doctor.add_argument(
        "--smoke",
        action="store_true",
        help="With --profile, also run a full pipeline smoke test",
    )
    doctor.set_defaults(func=_cmd_doctor)

    smoke = sub.add_parser("smoke", help="Run pipeline smoke test for one profile")
    smoke.add_argument("--profile", required=True)
    smoke.set_defaults(func=_cmd_smoke)

    bench = sub.add_parser("benchmark", help="Measure per-stage inference latency")
    bench.add_argument("--profile", default="quality")
    bench.add_argument("--runs", type=int, default=5)
    bench.add_argument("--images", default="tests/fixtures/images")
    bench.add_argument("--all", action="store_true", help="Benchmark multiple profiles")
    bench.add_argument("--profiles", nargs="+", help="Profiles when using --all")
    bench.add_argument("--output", help="JSON output path when using --all")
    bench.set_defaults(func=_cmd_benchmark)

    export = sub.add_parser("export", help="Export models to ONNX (maintainer)")
    export.add_argument("--output", default="output/onnx")
    export.add_argument("--skip-quantize", action="store_true")

    def _export_cmd(args: argparse.Namespace) -> int:
        argv = ["--output", args.output]
        if args.skip_quantize:
            argv.append("--skip-quantize")
        return export_main(argv)

    export.set_defaults(func=_export_cmd)

    val = sub.add_parser("validate", help="Validate ONNX vs PyTorch (maintainer)")
    val.add_argument("--models-dir", default="output/onnx")
    val.add_argument("--fixtures", default="tests/fixtures/images")

    def _validate_cmd(args: argparse.Namespace) -> int:
        return validate_main(
            ["--models-dir", args.models_dir, "--fixtures", args.fixtures]
        )

    val.set_defaults(func=_validate_cmd)

    eval_cmd = sub.add_parser("eval", help="Evaluate relevance on fixture subset")
    eval_cmd.add_argument("--profile", default="quality")
    eval_cmd.add_argument("--fixtures", default="tests/fixtures/eval")

    def _eval_cmd(args: argparse.Namespace) -> int:
        return run_eval(args.profile, ROOT / args.fixtures)

    eval_cmd.set_defaults(func=_eval_cmd)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
