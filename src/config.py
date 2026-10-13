"""Load YAML configuration files from ``configs/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from . import CONFIG_DIR


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_models(path: str | Path | None = None) -> Dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "models.yaml")


def load_tasks(path: str | Path | None = None) -> Dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "tasks.yaml")
