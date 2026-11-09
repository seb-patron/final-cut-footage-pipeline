"""Command-line interface for the Final Cut footage pipeline (skeleton).

The steps live in one ordered registry, :data:`STEPS`. A step whose handler
is None is not built yet: invoking it reads and writes nothing, prints
``<step>: not built yet`` to stderr and exits 3. ``run`` calls the built
main-path steps in registry order with the same ``--trip`` and ``--dry-run``,
never calls ingest, and exits 3 while no main-path step is built.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .net import OfflineError, set_offline

MAIN_PATH_STEPS = (
    "scan",
    "pull",
    "transcribe",
    "tag",
    "photos",
    "memo",
    "organize",
    "patch",
    "export",
    "roughcut",
)

OPTIONAL_STEP = "ingest"

NOT_BUILT_EXIT_CODE = 3

# A built step handler receives the trip folder and the dry-run flag and
# returns a process exit code.
Handler = Callable[[str, bool], int]


@dataclass(frozen=True)
class Step:
    """One pipeline step: its name, main-path membership and handler."""

    name: str
    main_path: bool
    handler: Handler | None


STEPS: list[Step] = [Step(name=name, main_path=True, handler=None) for name in MAIN_PATH_STEPS]
STEPS.append(Step(name=OPTIONAL_STEP, main_path=False, handler=None))


def _find_step(name: str) -> Step:
    for step in STEPS:
        if step.name == name:
            return step
    raise AssertionError(f"unknown step: {name}")


def run_step(step: Step, trip: str, dry_run: bool) -> int:
    """Run one step; unbuilt steps report and exit 3 without touching disk."""
    if step.handler is not None:
        return step.handler(trip, dry_run)
    print(f"{step.name}: not built yet", file=sys.stderr)
    return NOT_BUILT_EXIT_CODE


def run_command(trip: str, dry_run: bool) -> int:
    """Run the built main-path steps in registry order; never runs ingest."""
    ran_any = False
    for step in STEPS:
        if not step.main_path:
            continue
        if step.handler is None:
            print(f"run: skipping unbuilt step {step.name}", file=sys.stderr)
            continue
        ran_any = True
        code = step.handler(trip, dry_run)
        if code:
            return code
    if not ran_any:
        print("run: no steps built yet", file=sys.stderr)
        return NOT_BUILT_EXIT_CODE
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fcp_pipeline",
        description="Final Cut footage pipeline (skeleton: steps are unbuilt stubs).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="step")
    for step in STEPS:
        if step.main_path:
            help_text = f"run the {step.name} step (not built yet)"
        else:
            help_text = f"run the {step.name} step (optional, not built yet)"
        sub = subparsers.add_parser(step.name, help=help_text)
        sub.add_argument("--trip", required=True, help="trip folder holding project.toml")
        sub.add_argument("--dry-run", action="store_true", help="plan only; change nothing")
        sub.add_argument(
            "--offline",
            action="store_true",
            help="refuse every network call before connecting",
        )
    run_parser = subparsers.add_parser(
        "run", help="run the built main-path steps in order (never ingest)"
    )
    run_parser.add_argument("--trip", required=True, help="trip folder holding project.toml")
    run_parser.add_argument("--dry-run", action="store_true", help="plan only; change nothing")
    run_parser.add_argument(
        "--offline",
        action="store_true",
        help="refuse every network call before connecting",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "offline", False):
        set_offline(True)
    try:
        if args.command == "run":
            return run_command(args.trip, args.dry_run)
        return run_step(_find_step(args.command), args.trip, args.dry_run)
    except OfflineError as exc:
        # A handler refused a network call while offline: report one line
        # on stderr and exit 3, never a traceback.
        print(" ".join(str(exc).split()), file=sys.stderr)
        return NOT_BUILT_EXIT_CODE
