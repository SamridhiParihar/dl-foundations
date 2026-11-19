"""Absolute paths anchored to the repository root.

Relative paths like "data" or "checkpoints/x.pth" are resolved against the
directory you run the command FROM, so they silently change meaning when you
run a script from another folder. Build paths from these constants instead.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
