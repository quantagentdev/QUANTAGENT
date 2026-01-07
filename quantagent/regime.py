"""
MarketRegimeDetector - Market Regime Detection Module

Identifies and classifies market regimes (trending, mean-reverting, volatile)
for adaptive strategy selection.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    WEAK_BULL = "weak_bull"
    NEUTRAL = "neutral"
    WEAK_BEAR = "weak_bear"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MEAN_REVERTING = "mean_reverting"
    TRENDING = "trending"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"


class VolumeRegime(Enum):
    """Volume regime classifications."""
    HIGH_VOLUME = "high_volume"
    NORMAL_VOLUME = "normal_volume"
    LOW_VOLUME = "low_volume"
    CLIMAX = "climax"
    DRY_UP = "dry_up"


@dataclass
class RegimeState:
    """Current regime state with metadata."""
    primary_regime: MarketRegime
    secondary_regime: Optional[MarketRegime]
    volume_regime: VolumeRegime
    confidence: float
    duration_periods: int
    trend_strength: float
    volatility_percentile: float
    momentum_score: float
    mean_reversion_score: float
    detected_at: datetime


class MarketRegimeDetector:
    """
    Market Regime Detection Engine
    
    Detects and classifies market regimes using multiple methodologies:
    - Trend analysis (moving average alignment, ADX)
    - Volatility regime detection (historical vol percentiles)
    - Mean reversion scoring (distance from mean, RSI extremes)
    - Volume regime analysis
    - Hidden Markov Model-inspired state transitions
    
    Example:
        >>> detector = MarketRegimeDetector()
        >>> regime = detector.detect_regime(
        ...     prices=price_array,
        ...     volume=volume_array
        ... )
        >>> print(regime.primary_regime)
    """
    
    def __init__(
        self,
        lookback_short: int = 20,
        lookback_medium: int = 50,
        lookback_long: int = 200,
        volatility_window: int = 20,
        regime_change_threshold: float = 0.3,
    ):
        """
        Initialize the Market Regime Detector.
        
        Args:
            lookback_short: Short-term lookback period
            lookback_medium: Medium-term lookback period
            lookback_long: Long-term lookback period
            volatility_window: Window for volatility calculations
            regime_change_threshold: Threshold for detecting regime changes
        """
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long
        self.volatility_window = volatility_window
        self.regime_change_threshold = regime_change_threshold
        
        # Regime history for tracking transitions
        self.regime_history: List[RegimeState] = []

    def detect_regime(
        self,
        prices: np.ndarray,
        volume: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
    ) -> RegimeState:
        """
        Detect current market regime.
        
        Args:
            prices: Array of closing prices
            volume: Array of volume data (optional)
            high: Array of high prices (optional)
            low: Array of low prices (optional)
            
        Returns:
            RegimeState object with regime classification
        """
        # Calculate components
        trend_analysis = self._analyze_trend(prices)
        volatility_analysis = self._analyze_volatility(prices, high, low)
        momentum_analysis = self._analyze_momentum(prices)
        mean_reversion = self._analyze_mean_reversion(prices)
        volume_analysis = self._analyze_volume_regime(volume) if volume is not None else None
        
        # Determine primary regime
        primary_regime = self._classify_primary_regime(
            trend_analysis,
            volatility_analysis,
            momentum_analysis,
        )
        
        # Determine secondary regime (e.g., volatility overlay)
        secondary_regime = self._classify_secondary_regime(
            volatility_analysis,
            mean_reversion,
        )
        
        # Volume regime
        volume_regime = volume_analysis["regime"] if volume_analysis else VolumeRegime.NORMAL_VOLUME
        
        # Calculate overall confidence
        confidence = self._calculate_regime_confidence(
            trend_analysis,
            volatility_analysis,
            momentum_analysis,
        )
        
        # Create regime state
        regime_state = RegimeState(
            primary_regime=primary_regime,
            secondary_regime=secondary_regime,
            volume_regime=volume_regime,
            confidence=confidence,
            duration_periods=self._calculate_regime_duration(primary_regime),
            trend_strength=trend_analysis["strength"],
            volatility_percentile=volatility_analysis["percentile"],
            momentum_score=momentum_analysis["score"],
            mean_reversion_score=mean_reversion["score"],
            detected_at=datetime.now(),
        )
        
        # Track history
        self.regime_history.append(regime_state)
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        return regime_state

    def _analyze_trend(self, prices: np.ndarray) -> Dict[str, Any]:
        """Analyze trend characteristics."""
        if len(prices) < self.lookback_long:
            return {"direction": "neutral", "strength": 0.0, "aligned": False}
        
        # Moving averages
        sma_short = np.mean(prices[-self.lookback_short:])
        sma_medium = np.mean(prices[-self.lookback_medium:])
        sma_long = np.mean(prices[-self.lookback_long:])
        current = prices[-1]
        
        # Trend direction
        if current > sma_short > sma_medium > sma_long:
            direction = "strong_bull"
            aligned = True
        elif current > sma_short > sma_medium:
            direction = "bull"
            aligned = False
        elif current > sma_short:
            direction = "weak_bull"
            aligned = False
        elif current < sma_short < sma_medium < sma_long:
            direction = "strong_bear"
            aligned = True
        elif current < sma_short < sma_medium:
            direction = "bear"
            aligned = False
        elif current < sma_short:
            direction = "weak_bear"
            aligned = False
        else:
            direction = "neutral"
            aligned = False
        
        # Trend strength using linear regression slope
        x = np.arange(self.lookback_medium)
        recent_prices = prices[-self.lookback_medium:]
        slope = np.polyfit(x, recent_prices, 1)[0]
        strength = min(1.0, abs(slope) / np.std(recent_prices) * 10) if np.std(recent_prices) > 0 else 0.0
        
        # ADX proxy (simplified)
        returns = np.diff(prices[-self.lookback_medium:]) / prices[-self.lookback_medium:-1]
        adx_proxy = min(1.0, np.abs(np.mean(returns)) / np.std(returns) * 5) if np.std(returns) > 0 else 0.0
        
        return {
            "direction": direction,
            "strength": float(strength),
            "aligned": aligned,
            "adx_proxy": float(adx_proxy),
            "sma_short": float(sma_short),
            "sma_medium": float(sma_medium),
            "sma_long": float(sma_long),
        }

    def _analyze_volatility(
        self,
        prices: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Analyze volatility characteristics."""
        if len(prices) < self.volatility_window * 2:
            return {"regime": "normal", "percentile": 50.0, "current": 0.0}
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Current volatility
        current_vol = np.std(returns[-self.volatility_window:]) * np.sqrt(252)
        
        # Historical volatility distribution
        historical_vols = []
        for i in range(self.volatility_window, len(returns)):
            vol = np.std(returns[i-self.volatility_window:i]) * np.sqrt(252)
            historical_vols.append(vol)
        
        # Percentile
        if historical_vols:
            percentile = np.sum(np.array(historical_vols) <= current_vol) / len(historical_vols) * 100
        else:
            percentile = 50.0
        
        # Regime classification
        if percentile >= 80:
            regime = "high"
        elif percentile >= 60:
            regime = "above_average"
        elif percentile <= 20:
            regime = "low"
        elif percentile <= 40:
            regime = "below_average"
        else:
            regime = "normal"
        
        # ATR-based volatility if high/low available
        atr_pct = None
        if high is not None and low is not None and len(high) >= self.volatility_window:
            h = high[-self.volatility_window:]
            l = low[-self.volatility_window:]
            c_prev = prices[-self.volatility_window-1:-1] if len(prices) > self.volatility_window else prices[-self.volatility_window:]
            
            tr = h - l  # Simplified true range
            atr = np.mean(tr)
            atr_pct = atr / prices[-1] * 100 if prices[-1] > 0 else None
        
        return {
            "regime": regime,
            "percentile": float(percentile),
            "current": float(current_vol),
            "atr_pct": float(atr_pct) if atr_pct else None,
        }

    def _analyze_momentum(self, prices: np.ndarray) -> Dict[str, Any]:
        """Analyze momentum characteristics."""
        if len(prices) < self.lookback_medium:
            return {"score": 0.0, "rsi": 50.0, "roc": 0.0}
        
        # Rate of change
        roc_short = (prices[-1] - prices[-self.lookback_short]) / prices[-self.lookback_short]
        roc_medium = (prices[-1] - prices[-self.lookback_medium]) / prices[-self.lookback_medium]
        
        # RSI calculation
        deltas = np.diff(prices[-self.lookback_short-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Momentum score (-1 to 1)
        momentum_score = (rsi - 50) / 50  # -1 to 1 scale
        
        return {
            "score": float(momentum_score),
            "rsi": float(rsi),
            "roc_short": float(roc_short),
            "roc_medium": float(roc_medium),
        }

    def _analyze_mean_reversion(self, prices: np.ndarray) -> Dict[str, Any]:
        """Analyze mean reversion characteristics."""
        if len(prices) < self.lookback_medium:
            return {"score": 0.0, "z_score": 0.0, "distance_from_mean": 0.0}
        
        # Z-score from mean
        mean = np.mean(prices[-self.lookback_medium:])
        std = np.std(prices[-self.lookback_medium:])
        
        if std == 0:
            z_score = 0.0
        else:
            z_score = (prices[-1] - mean) / std
        
        # Distance from mean
        distance = (prices[-1] - mean) / mean * 100
        
        # Mean reversion score (higher = more likely to revert)
        # Extreme z-scores suggest mean reversion opportunity
        mr_score = min(1.0, abs(z_score) / 2)  # Cap at |z| = 2
        
        return {
            "score": float(mr_score),
            "z_score": float(z_score),
            "distance_from_mean": float(distance),
        }

    def _analyze_volume_regime(self, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze volume regime."""
        if len(volume) < self.lookback_short:
            return {"regime": VolumeRegime.NORMAL_VOLUME, "ratio": 1.0}
        
        avg_volume = np.mean(volume[-self.lookback_medium:])
        recent_avg = np.mean(volume[-5:])
        current = volume[-1]
        
        volume_ratio = current / avg_volume if avg_volume > 0 else 1.0
        recent_ratio = recent_avg / avg_volume if avg_volume > 0 else 1.0
        
        # Classify volume regime
        if volume_ratio > 3.0:
            regime = VolumeRegime.CLIMAX
        elif volume_ratio > 1.5:
            regime = VolumeRegime.HIGH_VOLUME
        elif volume_ratio < 0.3:
            regime = VolumeRegime.DRY_UP
        elif volume_ratio < 0.7:
            regime = VolumeRegime.LOW_VOLUME
        else:
            regime = VolumeRegime.NORMAL_VOLUME
        
        return {
            "regime": regime,
            "ratio": float(volume_ratio),
            "recent_ratio": float(recent_ratio),
        }

    def _classify_primary_regime(
        self,
        trend: Dict[str, Any],
        volatility: Dict[str, Any],
        momentum: Dict[str, Any],
    ) -> MarketRegime:
        """Classify primary market regime."""
        trend_dir = trend["direction"]
        trend_strength = trend["strength"]
        vol_regime = volatility["regime"]
        mom_score = momentum["score"]
        
        # High volatility overrides
        if vol_regime == "high" and volatility["percentile"] > 90:
            return MarketRegime.HIGH_VOLATILITY
        
        # Low volatility regime
        if vol_regime == "low" and volatility["percentile"] < 10:
            return MarketRegime.LOW_VOLATILITY
        
        # Trend-based classification
        if trend_dir == "strong_bull" and trend["aligned"]:
            return MarketRegime.STRONG_BULL
        elif trend_dir in ["strong_bull", "bull"] and trend_strength > 0.6:
            return MarketRegime.BULL
        elif trend_dir in ["bull", "weak_bull"]:
            return MarketRegime.WEAK_BULL
        elif trend_dir == "strong_bear" and trend["aligned"]:
            return MarketRegime.STRONG_BEAR
        elif trend_dir in ["strong_bear", "bear"] and trend_strength > 0.6:
            return MarketRegime.BEAR
        elif trend_dir in ["bear", "weak_bear"]:
            return MarketRegime.WEAK_BEAR
        else:
            # Neutral - check for consolidation or mean reversion
            if trend_strength < 0.2 and abs(mom_score) < 0.3:
                return MarketRegime.CONSOLIDATION
            return MarketRegime.NEUTRAL

    def _classify_secondary_regime(
        self,
        volatility: Dict[str, Any],
        mean_reversion: Dict[str, Any],
    ) -> Optional[MarketRegime]:
        """Classify secondary/overlay regime."""
        # Mean reversion potential
        if mean_reversion["score"] > 0.7:
            return MarketRegime.MEAN_REVERTING
        
        # Trending overlay
        if volatility["percentile"] > 60:
            return MarketRegime.TRENDING
        
        # Breakout potential (low vol + extreme mean reversion)
        if volatility["percentile"] < 30 and mean_reversion["score"] > 0.5:
            return MarketRegime.BREAKOUT
        
        return None

    def _calculate_regime_confidence(
        self,
        trend: Dict[str, Any],
        volatility: Dict[str, Any],
        momentum: Dict[str, Any],
    ) -> float:
        """Calculate confidence in regime classification."""
        confidence_factors = []
        
        # Trend alignment
        if trend["aligned"]:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5 + trend["strength"] * 0.3)
        
        # Volatility regime clarity
        vol_pct = volatility["percentile"]
        if vol_pct > 80 or vol_pct < 20:
            confidence_factors.append(0.85)
        elif vol_pct > 60 or vol_pct < 40:
            confidence_factors.append(0.65)
        else:
            confidence_factors.append(0.5)
        
        # Momentum alignment
        mom_abs = abs(momentum["score"])
        confidence_factors.append(0.5 + mom_abs * 0.4)
        
        return float(np.mean(confidence_factors))

    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long the current regime has been in effect."""
        if not self.regime_history:
            return 1
        
        duration = 1
        for state in reversed(self.regime_history[:-1]):
            if state.primary_regime == current_regime:
                duration += 1
            else:
                break
        
        return duration

    def get_regime_transitions(self) -> List[Dict[str, Any]]:
        """Get history of regime transitions."""
        if len(self.regime_history) < 2:
            return []
        
        transitions = []
        for i in range(1, len(self.regime_history)):
            prev = self.regime_history[i-1]
            curr = self.regime_history[i]
            
            if prev.primary_regime != curr.primary_regime:
                transitions.append({
                    "from": prev.primary_regime.value,
                    "to": curr.primary_regime.value,
                    "timestamp": curr.detected_at.isoformat(),
                    "confidence": curr.confidence,
                })
        
        return transitions

    def get_optimal_strategy(self, regime: RegimeState) -> Dict[str, Any]:
        """
        Suggest optimal trading strategy for current regime.
        
        Args:
            regime: Current regime state
            
        Returns:
            Strategy recommendations dictionary
        """
        strategies = {
            MarketRegime.STRONG_BULL: {
                "bias": "long",
                "approach": "trend_following",
                "position_sizing": "aggressive",
                "stop_type": "trailing",
                "indicators": ["moving_averages", "momentum"],
            },
            MarketRegime.BULL: {
                "bias": "long",
                "approach": "trend_following",
                "position_sizing": "normal",
                "stop_type": "trailing",
                "indicators": ["moving_averages", "volume"],
            },
            MarketRegime.WEAK_BULL: {
                "bias": "neutral_to_long",
                "approach": "selective_breakouts",
                "position_sizing": "reduced",
                "stop_type": "fixed",
                "indicators": ["support_resistance", "volume"],
            },
            MarketRegime.STRONG_BEAR: {
                "bias": "short",
                "approach": "trend_following",
                "position_sizing": "aggressive",
                "stop_type": "trailing",
                "indicators": ["moving_averages", "momentum"],
            },
            MarketRegime.BEAR: {
                "bias": "short",
                "approach": "trend_following",
                "position_sizing": "normal",
                "stop_type": "trailing",
                "indicators": ["moving_averages", "volume"],
            },
            MarketRegime.WEAK_BEAR: {
                "bias": "neutral_to_short",
                "approach": "selective_breakouts",
                "position_sizing": "reduced",
                "stop_type": "fixed",
                "indicators": ["support_resistance", "volume"],
            },
            MarketRegime.NEUTRAL: {
                "bias": "neutral",
                "approach": "range_trading",
                "position_sizing": "small",
                "stop_type": "fixed",
                "indicators": ["support_resistance", "oscillators"],
            },
            MarketRegime.CONSOLIDATION: {
                "bias": "neutral",
                "approach": "wait_for_breakout",
                "position_sizing": "minimal",
                "stop_type": "volatility_based",
                "indicators": ["bollinger_bands", "volume"],
            },
            MarketRegime.HIGH_VOLATILITY: {
                "bias": "neutral",
                "approach": "volatility_harvesting",
                "position_sizing": "reduced",
                "stop_type": "wide_fixed",
                "indicators": ["atr", "vix"],
            },
            MarketRegime.LOW_VOLATILITY: {
                "bias": "neutral",
                "approach": "breakout_anticipation",
                "position_sizing": "normal",
                "stop_type": "tight_fixed",
                "indicators": ["bollinger_squeeze", "volume"],
            },
        }
        
        strategy = strategies.get(regime.primary_regime, strategies[MarketRegime.NEUTRAL])
        
        # Adjust for secondary regime
        if regime.secondary_regime == MarketRegime.MEAN_REVERTING:
            strategy["approach"] = "mean_reversion"
            strategy["indicators"].append("rsi")
        elif regime.secondary_regime == MarketRegime.BREAKOUT:
            strategy["approach"] = "breakout"
            strategy["indicators"].append("volume_profile")
        
        return {
            "regime": regime.primary_regime.value,
            "secondary_regime": regime.secondary_regime.value if regime.secondary_regime else None,
            "confidence": regime.confidence,
            "strategy": strategy,
        }

