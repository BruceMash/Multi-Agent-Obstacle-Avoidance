"""Expose the local MARL directory through the original marllib.marl path."""

from pathlib import Path

_MARL_ROOT = Path(__file__).resolve().parents[2] / "MARL"
__path__ = [str(_MARL_ROOT)]

