from __future__ import annotations

from pathlib import Path

import pytest
import torch

from isac_verify.config import PilotConfig


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config(repo_root: Path) -> PilotConfig:
    return PilotConfig.load(repo_root / "configs" / "pilot.yaml")


@pytest.fixture(scope="session")
def output_dir(repo_root: Path) -> Path:
    path = repo_root / "results" / "phase2"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session", autouse=True)
def require_cuda() -> None:
    assert torch.cuda.is_available(), "CUDA is required; CPU fallback is forbidden"
