"""Fuzzy-logic student performance prediction (case study)."""
from .engine import MamdaniFIS, Rule, defuzzify
from .model import FISConfig, build_fis

__all__ = ["MamdaniFIS", "Rule", "defuzzify", "FISConfig", "build_fis"]
