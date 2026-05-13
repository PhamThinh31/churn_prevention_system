"""Smoke tests — keep CI green while early-phase modules stabilize."""


def test_imports():
    from src.data import labels, loader, splits  # noqa: F401
