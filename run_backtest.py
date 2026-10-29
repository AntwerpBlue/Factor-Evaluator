#!/usr/bin/env python3
"""The single executable entry point for Factor Evaluator."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from factor_evaluator.cli import main


if __name__ == "__main__":
    main()
