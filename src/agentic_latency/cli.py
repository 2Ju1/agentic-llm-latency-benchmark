from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_latency.config import load_config
from agentic_latency.plan import build_plan
from agentic_latency.runner import execute_plan, summarize_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-latency",
        description="Inspect reproducible agentic LLM latency experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="Render the plan without requests.")
    plan_parser.add_argument("--config", default="configs/default.yaml", type=Path)
    plan_parser.add_argument("--json", action="store_true", help="Print the full plan as JSON.")

    run_parser = subparsers.add_parser("run", help="Execute the configured experiment matrix.")
    run_parser.add_argument("--config", default="configs/default.yaml", type=Path)
    run_parser.add_argument("--output", default="outputs/results_raw.csv", type=Path)
    run_parser.add_argument("--no-warmup", action="store_true")

    summary_parser = subparsers.add_parser("summarize", help="Aggregate a raw result CSV.")
    summary_parser.add_argument("input", type=Path)
    summary_parser.add_argument("--output", default="outputs/summary.csv", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plan":
        config = load_config(args.config)
        plan = build_plan(config)
        if args.json:
            print(json.dumps([case.to_dict() for case in plan], indent=2))
            return
        counts: dict[str, int] = {}
        for case in plan:
            counts[case.experiment] = counts.get(case.experiment, 0) + 1
        print(f"model: {config.model}")
        print(f"requests: {len(plan)}")
        for experiment, count in counts.items():
            print(f"- {experiment}: {count}")
    elif args.command == "run":
        config = load_config(args.config)
        execute_plan(config, args.output, warmup=not args.no_warmup)
        summary_path = args.output.with_name("summary.csv")
        summarize_results(args.output, summary_path)
        print(f"raw results: {args.output}")
        print(f"summary: {summary_path}")
    elif args.command == "summarize":
        summary = summarize_results(args.input, args.output)
        print(summary.to_string(index=False))
        print(f"summary: {args.output}")


if __name__ == "__main__":
    main()

