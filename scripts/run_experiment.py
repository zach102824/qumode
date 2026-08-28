#!/usr/bin/env python3
"""Thin wrapper so ``python scripts/run_experiment.py`` works after editable install."""

from qumode_vqe.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
