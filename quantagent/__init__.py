"""
$QUANTAGENT - Elite Quantitative Trading Analysis

Usage:
    python -m quantagent analyze AAPL
    python -m quantagent analyze BTC-USD
    python -m quantagent analyze BTC-USD --ai   # With AI insights (requires swarms)

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "QuantAgent Team"

# Core modules (no external dependencies beyond numpy/scipy)
from quantagent.technical import TechnicalEngine
from quantagent.fundamental import FundamentalEngine
from quantagent.risk import RiskEngine
from quantagent.targets import PriceTargetGenerator
from quantagent.regime import MarketRegimeDetector
from quantagent.signals import SignalAggregator

__all__ = [
    "TechnicalEngine",
    "FundamentalEngine",
    "RiskEngine",
    "PriceTargetGenerator",
    "MarketRegimeDetector",
    "SignalAggregator",
    "QuantAgent",
]

# Lazy import for AI agent (requires swarms)
def __getattr__(name):
    if name == "QuantAgent":
        from quantagent.agent import QuantAgent
        return QuantAgent
    if name == "QuantTraderAgent":
        from quantagent.core import QuantTraderAgent
        return QuantTraderAgent
    raise AttributeError(f"module 'quantagent' has no attribute '{name}'")
