"""Shared pytest fixtures for bandaid."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HK13_SESSION = REPO_ROOT / "data" / "HK13" / "HK13_20260814"


@pytest.fixture(scope="session")
def hk13_session() -> Path:
    """Path to the HK13 example session; skips the test when the data is absent."""
    if not HK13_SESSION.is_dir() or not any(HK13_SESSION.glob("*.tif")):
        pytest.skip(f"example data not available at {HK13_SESSION}")
    return HK13_SESSION


@pytest.fixture(scope="session")
def hk13_tiffs(hk13_session: Path) -> list[Path]:
    return sorted(hk13_session.glob("*.tif"))


@pytest.fixture(scope="session")
def synthetic_movie():
    from synthetic import make_synthetic_movie

    return make_synthetic_movie()


@pytest.fixture(scope="session")
def no_ripple_movie():
    from synthetic import make_synthetic_movie

    return make_synthetic_movie(with_ripple=False, seed=1)
