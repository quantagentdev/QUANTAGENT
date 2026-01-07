"""
SignalAggregator - Signal Aggregation Module

Combines signals from multiple sources (technical, fundamental, sentiment)
into actionable trading signals with confidence scoring.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal type classifications."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class SignalSource(Enum):
    """Signal source categories."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    TREND = "trend"
    PATTERN = "pattern"
    VOLATILITY = "volatility"


@dataclass
class Signal:
    """Individual signal from a source."""
    source: SignalSource
    name: str
    value: float  # -1 to 1 scale
    weight: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedSignal:
    """Aggregated signal from multiple sources."""
    signal_type: SignalType
    composite_score: float
    confidence: float
    conviction: str  # 'high', 'medium', 'low'
    contributing_signals: List[Signal]
    conflicting_signals: List[Signal]
    agreement_pct: float
    timestamp: datetime = field(default_factory=datetime.now)


class SignalAggregator:
    """
    Signal Aggregation Engine
    
    Combines multiple signal sources into unified trading signals:
    - Weighted signal combination
    - Conflict detection and resolution
    - Confidence adjustment based on agreement
    - Multi-timeframe signal confluence
    - Dynamic weight adjustment based on market regime
    
    Example:
        >>> aggregator = SignalAggregator()
        >>> aggregator.add_signal(
        ...     source=SignalSource.TECHNICAL,
        ...     name="RSI",
        ...     value=0.6,
        ...     confidence=0.8
        ... )
        >>> result = aggregator.aggregate()
    """
    
    # Default signal weights
    DEFAULT_WEIGHTS = {
        SignalSource.TREND: 0.20,
        SignalSource.MOMENTUM: 0.18,
        SignalSource.TECHNICAL: 0.15,
        SignalSource.VOLUME: 0.12,
        SignalSource.PATTERN: 0.12,
        SignalSource.FUNDAMENTAL: 0.13,
        SignalSource.SENTIMENT: 0.05,
        SignalSource.VOLATILITY: 0.05,
    }
    
    def __init__(
        self,
        weights: Optional[Dict[SignalSource, float]] = None,
        conflict_threshold: float = 0.4,
        high_conviction_threshold: float = 0.7,
    ):
        """
        Initialize the Signal Aggregator.
        
        Args:
            weights: Custom weights for signal sources
            conflict_threshold: Threshold for detecting conflicting signals
            high_conviction_threshold: Threshold for high conviction signals
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.conflict_threshold = conflict_threshold
        self.high_conviction_threshold = high_conviction_threshold
        
        # Signal storage
        self.signals: List[Signal] = []
        self.signal_history: List[AggregatedSignal] = []

    def add_signal(
        self,
        source: SignalSource,
        name: str,
        value: float,
        confidence: float = 1.0,
        weight_override: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a signal to the aggregator.
        
        Args:
            source: Signal source category
            name: Signal name/identifier
            value: Signal value (-1 to 1, negative=bearish, positive=bullish)
            confidence: Signal confidence (0 to 1)
            weight_override: Override default source weight
            metadata: Additional signal metadata
        """
        # Clamp value to valid range
        value = max(-1.0, min(1.0, value))
        confidence = max(0.0, min(1.0, confidence))
        
        weight = weight_override if weight_override is not None else self.weights.get(source, 0.1)
        
        signal = Signal(
            source=source,
            name=name,
            value=value,
            weight=weight,
            confidence=confidence,
            metadata=metadata or {},
        )
        
        self.signals.append(signal)
        logger.debug(f"Added signal: {name} = {value:.2f} (conf: {confidence:.2f})")

    def add_technical_signals(
        self,
        rsi: Optional[float] = None,
        macd_signal: Optional[float] = None,
        moving_average_signal: Optional[float] = None,
        bollinger_signal: Optional[float] = None,
        stochastic_signal: Optional[float] = None,
    ) -> None:
        """Add common technical indicator signals."""
        
        if rsi is not None:
            # Convert RSI (0-100) to signal (-1 to 1)
            if rsi > 70:
                value = -(rsi - 70) / 30  # Overbought = bearish
            elif rsi < 30:
                value = (30 - rsi) / 30   # Oversold = bullish
            else:
                value = (rsi - 50) / 40   # Neutral zone
            
            self.add_signal(
                source=SignalSource.MOMENTUM,
                name="RSI",
                value=value,
                confidence=0.75,
                metadata={"raw_value": rsi},
            )
        
        if macd_signal is not None:
            self.add_signal(
                source=SignalSource.MOMENTUM,
                name="MACD",
                value=macd_signal,
                confidence=0.70,
            )
        
        if moving_average_signal is not None:
            self.add_signal(
                source=SignalSource.TREND,
                name="Moving_Averages",
                value=moving_average_signal,
                confidence=0.80,
            )
        
        if bollinger_signal is not None:
            self.add_signal(
                source=SignalSource.VOLATILITY,
                name="Bollinger_Bands",
                value=bollinger_signal,
                confidence=0.65,
            )
        
        if stochastic_signal is not None:
            self.add_signal(
                source=SignalSource.MOMENTUM,
                name="Stochastic",
                value=stochastic_signal,
                confidence=0.65,
            )

    def add_fundamental_signal(
        self,
        fair_value_upside: float,
        confidence: float = 0.7,
    ) -> None:
        """
        Add fundamental valuation signal.
        
        Args:
            fair_value_upside: Percentage upside to fair value
            confidence: Confidence in the valuation
        """
        # Convert upside percentage to signal
        # Cap at +/- 50% for normalization
        normalized = max(-50, min(50, fair_value_upside)) / 50
        
        self.add_signal(
            source=SignalSource.FUNDAMENTAL,
            name="Fair_Value",
            value=normalized,
            confidence=confidence,
            metadata={"upside_pct": fair_value_upside},
        )

    def add_sentiment_signal(
        self,
        sentiment_score: float,
        confidence: float = 0.6,
    ) -> None:
        """
        Add market sentiment signal.
        
        Args:
            sentiment_score: Sentiment score (-1 to 1)
            confidence: Confidence in sentiment reading
        """
        self.add_signal(
            source=SignalSource.SENTIMENT,
            name="Market_Sentiment",
            value=sentiment_score,
            confidence=confidence,
        )

    def add_volume_signal(
        self,
        volume_signal: float,
        confidence: float = 0.7,
    ) -> None:
        """
        Add volume-based signal.
        
        Args:
            volume_signal: Volume signal (-1 to 1)
            confidence: Confidence in volume analysis
        """
        self.add_signal(
            source=SignalSource.VOLUME,
            name="Volume_Analysis",
            value=volume_signal,
            confidence=confidence,
        )

    def aggregate(
        self,
        require_confluence: bool = False,
        min_signals: int = 2,
    ) -> AggregatedSignal:
        """
        Aggregate all signals into a unified signal.
        
        Args:
            require_confluence: Only generate signal if signals agree
            min_signals: Minimum number of signals required
            
        Returns:
            AggregatedSignal object
        """
        if len(self.signals) < min_signals:
            return AggregatedSignal(
                signal_type=SignalType.NEUTRAL,
                composite_score=0.0,
                confidence=0.0,
                conviction="low",
                contributing_signals=[],
                conflicting_signals=[],
                agreement_pct=0.0,
            )
        
        # Calculate weighted score
        total_weight = 0.0
        weighted_sum = 0.0
        
        for signal in self.signals:
            adjusted_weight = signal.weight * signal.confidence
            weighted_sum += signal.value * adjusted_weight
            total_weight += adjusted_weight
        
        composite_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Detect conflicts
        bullish_signals = [s for s in self.signals if s.value > 0.2]
        bearish_signals = [s for s in self.signals if s.value < -0.2]
        
        # Calculate agreement
        if len(self.signals) > 0:
            same_direction = sum(1 for s in self.signals if np.sign(s.value) == np.sign(composite_score))
            agreement_pct = same_direction / len(self.signals) * 100
        else:
            agreement_pct = 0.0
        
        # Identify conflicting signals
        if composite_score > 0:
            conflicting = bearish_signals
            contributing = bullish_signals
        else:
            conflicting = bullish_signals
            contributing = bearish_signals
        
        # Check for significant conflicts
        has_major_conflict = any(
            abs(s.value - composite_score) > self.conflict_threshold
            for s in self.signals
        )
        
        # Adjust confidence based on agreement
        base_confidence = np.mean([s.confidence for s in self.signals])
        agreement_factor = agreement_pct / 100
        confidence = base_confidence * (0.5 + 0.5 * agreement_factor)
        
        if has_major_conflict:
            confidence *= 0.8
        
        # Determine conviction
        if abs(composite_score) > self.high_conviction_threshold and agreement_pct > 70:
            conviction = "high"
        elif abs(composite_score) > 0.4 and agreement_pct > 50:
            conviction = "medium"
        else:
            conviction = "low"
        
        # Require confluence check
        if require_confluence and agreement_pct < 60:
            composite_score = 0.0
            conviction = "low"
        
        # Classify signal type
        signal_type = self._classify_signal(composite_score, conviction)
        
        result = AggregatedSignal(
            signal_type=signal_type,
            composite_score=float(composite_score),
            confidence=float(confidence),
            conviction=conviction,
            contributing_signals=contributing,
            conflicting_signals=conflicting,
            agreement_pct=float(agreement_pct),
        )
        
        # Store in history
        self.signal_history.append(result)
        
        return result

    def _classify_signal(
        self,
        score: float,
        conviction: str,
    ) -> SignalType:
        """Classify composite score into signal type."""
        if score >= 0.6 and conviction == "high":
            return SignalType.STRONG_BUY
        elif score >= 0.3:
            return SignalType.BUY
        elif score >= 0.1:
            return SignalType.WEAK_BUY
        elif score <= -0.6 and conviction == "high":
            return SignalType.STRONG_SELL
        elif score <= -0.3:
            return SignalType.SELL
        elif score <= -0.1:
            return SignalType.WEAK_SELL
        else:
            return SignalType.NEUTRAL

    def clear_signals(self) -> None:
        """Clear all current signals."""
        self.signals = []

    def get_signal_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of current signals."""
        if not self.signals:
            return {"error": "No signals added"}
        
        breakdown = {
            "total_signals": len(self.signals),
            "by_source": {},
            "bullish_count": sum(1 for s in self.signals if s.value > 0.1),
            "bearish_count": sum(1 for s in self.signals if s.value < -0.1),
            "neutral_count": sum(1 for s in self.signals if abs(s.value) <= 0.1),
        }
        
        for source in SignalSource:
            source_signals = [s for s in self.signals if s.source == source]
            if source_signals:
                breakdown["by_source"][source.value] = {
                    "count": len(source_signals),
                    "avg_value": float(np.mean([s.value for s in source_signals])),
                    "avg_confidence": float(np.mean([s.confidence for s in source_signals])),
                    "signals": [
                        {"name": s.name, "value": s.value, "confidence": s.confidence}
                        for s in source_signals
                    ],
                }
        
        return breakdown

    def adjust_weights_for_regime(
        self,
        regime: str,
    ) -> None:
        """
        Adjust signal weights based on market regime.
        
        Args:
            regime: Current market regime
        """
        regime_adjustments = {
            "strong_bull": {
                SignalSource.TREND: 1.3,
                SignalSource.MOMENTUM: 1.2,
                SignalSource.VOLUME: 1.1,
                SignalSource.VOLATILITY: 0.8,
            },
            "bull": {
                SignalSource.TREND: 1.2,
                SignalSource.MOMENTUM: 1.1,
            },
            "strong_bear": {
                SignalSource.TREND: 1.3,
                SignalSource.MOMENTUM: 1.2,
                SignalSource.VOLUME: 1.1,
                SignalSource.VOLATILITY: 0.8,
            },
            "bear": {
                SignalSource.TREND: 1.2,
                SignalSource.MOMENTUM: 1.1,
            },
            "high_volatility": {
                SignalSource.VOLATILITY: 1.5,
                SignalSource.TREND: 0.8,
                SignalSource.PATTERN: 0.7,
            },
            "low_volatility": {
                SignalSource.PATTERN: 1.3,
                SignalSource.VOLATILITY: 0.7,
            },
            "consolidation": {
                SignalSource.PATTERN: 1.3,
                SignalSource.VOLUME: 1.2,
                SignalSource.TREND: 0.8,
            },
            "mean_reverting": {
                SignalSource.MOMENTUM: 1.4,
                SignalSource.TREND: 0.7,
            },
        }
        
        adjustments = regime_adjustments.get(regime, {})
        
        for source, factor in adjustments.items():
            if source in self.weights:
                self.weights[source] *= factor
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def multi_timeframe_confluence(
        self,
        signals_by_timeframe: Dict[str, List[Signal]],
    ) -> Dict[str, Any]:
        """
        Analyze signal confluence across multiple timeframes.
        
        Args:
            signals_by_timeframe: Dictionary mapping timeframe to signals
            
        Returns:
            Confluence analysis dictionary
        """
        timeframe_scores = {}
        
        for tf, signals in signals_by_timeframe.items():
            if signals:
                avg_score = np.mean([s.value for s in signals])
                avg_confidence = np.mean([s.confidence for s in signals])
                timeframe_scores[tf] = {
                    "score": float(avg_score),
                    "confidence": float(avg_confidence),
                    "direction": "bullish" if avg_score > 0 else "bearish" if avg_score < 0 else "neutral",
                }
        
        # Check for alignment
        directions = [ts["direction"] for ts in timeframe_scores.values()]
        all_aligned = len(set(directions)) == 1 and "neutral" not in directions
        
        # Overall confluence score
        if all_aligned:
            confluence_score = np.mean([ts["score"] for ts in timeframe_scores.values()])
            confluence_confidence = 0.9
        else:
            # Weighted by typical importance (longer timeframes more weight)
            tf_weights = {"1h": 0.15, "4h": 0.25, "D1": 0.35, "W1": 0.25}
            weighted_score = 0
            total_weight = 0
            for tf, ts in timeframe_scores.items():
                weight = tf_weights.get(tf, 0.25)
                weighted_score += ts["score"] * weight
                total_weight += weight
            confluence_score = weighted_score / total_weight if total_weight > 0 else 0
            confluence_confidence = 0.5 + 0.2 * (1 - len(set(directions)) / 3)
        
        return {
            "aligned": all_aligned,
            "confluence_score": float(confluence_score),
            "confluence_confidence": float(confluence_confidence),
            "timeframe_details": timeframe_scores,
            "recommendation": self._classify_signal(confluence_score, "high" if all_aligned else "medium").value,
        }

    def generate_actionable_signal(
        self,
        current_price: float,
        atr: float,
    ) -> Dict[str, Any]:
        """
        Generate actionable trading signal with entry/exit parameters.
        
        Args:
            current_price: Current market price
            atr: Average True Range
            
        Returns:
            Actionable signal dictionary
        """
        aggregated = self.aggregate()
        
        if aggregated.signal_type == SignalType.NEUTRAL:
            return {
                "action": "HOLD",
                "signal_type": aggregated.signal_type.value,
                "confidence": aggregated.confidence,
                "reason": "No clear directional signal",
            }
        
        is_bullish = aggregated.composite_score > 0
        
        # Entry price (could be market or limit)
        if is_bullish:
            entry_price = current_price  # Market entry for simplicity
            stop_loss = current_price - 2 * atr
            target_1 = current_price + 1.5 * atr
            target_2 = current_price + 3 * atr
            target_3 = current_price + 5 * atr
            action = "BUY"
        else:
            entry_price = current_price
            stop_loss = current_price + 2 * atr
            target_1 = current_price - 1.5 * atr
            target_2 = current_price - 3 * atr
            target_3 = current_price - 5 * atr
            action = "SELL"
        
        # Risk/reward
        risk = abs(entry_price - stop_loss)
        reward_1 = abs(target_1 - entry_price)
        
        return {
            "action": action,
            "signal_type": aggregated.signal_type.value,
            "composite_score": aggregated.composite_score,
            "confidence": aggregated.confidence,
            "conviction": aggregated.conviction,
            "entry": {
                "price": round(entry_price, 2),
                "type": "market",
            },
            "stop_loss": round(stop_loss, 2),
            "targets": {
                "target_1": round(target_1, 2),
                "target_2": round(target_2, 2),
                "target_3": round(target_3, 2),
            },
            "risk_reward": round(reward_1 / risk, 2) if risk > 0 else 0,
            "agreement_pct": aggregated.agreement_pct,
            "bullish_signals": len(aggregated.contributing_signals) if is_bullish else len(aggregated.conflicting_signals),
            "bearish_signals": len(aggregated.conflicting_signals) if is_bullish else len(aggregated.contributing_signals),
        }

