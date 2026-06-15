"""
main.py
-------
Top-level entry point for ATSC-Detroit.

This is the only file a new user needs to run to operate the project.
All subcommands delegate to the appropriate module — nothing heavy lives here.

Usage:
    python main.py train
    python main.py evaluate --model models/best_model.zip
    python main.py evaluate --model models/best_model.zip --episodes 10
    python main.py baseline --episodes 5
    python main.py watch --model models/best_model.zip
    python main.py compare --model models/best_model.zip
"""

import sys
import argparse
from training.train import train
from training.evaluate import main as evaluate_main


def print_usage():
    """Print a friendly usage block when no subcommand is given."""

    print("""
    ATSC Detroit - Adaptive Traffic Signal Control
    \nUsage:
    \tpython main.py train\t\t\t\t\tTrain the PPO agent
    \tpython main.py evaluate\t\t--model <path>\t\tEvaluate
    \tpython main.py baseline\t\t--episodes <n>\t\tRun the fixed baseline
    \tpython main.py watch\t\t--model <path>\t\tWatch agent in SUMO GUI
    \tpython main.py compare\t\t--model <path>\t\tCompare agent vs baseline
    \nOptions:
    \t--model <path>\t\tPath to a saved model .zip file
    \t--episodes <n>\t\tNumber of episodes to run (default: 5)
    \t--gui\t\tLaunch SUMO with the visual interface
    """)


def handle_train(args):
    """Delegate to the training module."""
    train()


def handle_evaluate(args):
    """Delegate to the evaluation module."""
    sys.argv = ["evaluate.py", "--model", args.model, "--episodes", str(args.episodes)]

    if args.gui:
        sys.argv.append("--gui")
    evaluate_main()


def handle_baseline(args):
    """Run the fixed baseline via the evaluation module."""
    sys.argv = [
        "evaluate.py",
        "--baseline",
        "--episodes",
        str(args.episodes),
    ]
    evaluate_main()


def handle_watch(args):
    """Load a saved model and run one episode with the SUMO GUI."""
    sys.argv = ["evaluate.py", "--model", args.model, "--gui", "--episodes", "1"]
    evaluate_main()


def handle_compare(args):
    """Run both agents and print a side-by-side comparison."""
    sys.argv = [
        "evaluate.py",
        "--compare",
        "--model",
        args.model,
        "--episodes",
        str(args.episodes),
    ]

    evaluate_main()


def main():
    parser = argparse.ArgumentParser(description="ATSC-Detroit", add_help=False)

    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        choices=["train", "evaluate", "baseline", "watch", "compare"],
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--gui", action="store_true")

    args = parser.parse_args()
    if not args.command:
        print_usage()
        sys.exit(0)

    handlers = {
        "train": handle_train,
        "evaluate": handle_evaluate,
        "baseline": handle_baseline,
        "watch": handle_watch,
        "compare": handle_compare,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
