"""
**icecream** explains how a machine learning model works using
Partial Dependency Plots and various Individual Conditional Expectation plots
"""

from .config import options
from .icecream import IceCream, IceCream2D

__all__ = ["IceCream", "IceCream2D", "options"]
