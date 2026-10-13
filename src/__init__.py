"""AI6130 Assignment 1 - Open-LLM-Benchmark evaluation package."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
SCORES_DIR = RESULTS_DIR / "scores"
FIGURES_DIR = PROJECT_ROOT / "figures"

__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "RESULTS_DIR",
    "RAW_DIR",
    "SCORES_DIR",
    "FIGURES_DIR",
]
