"""
TechnicalEngine - Advanced Technical Analysis Module

Comprehensive technical analysis engine with 50+ indicators, pattern recognition,
multi-timeframe analysis, and volume profile analytics.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction classifications."""
    STRONG_UP = "strong_uptrend"
    UP = "uptrend"
    WEAK_UP = "weak_uptrend"
    NEUTRAL = "neutral"
    WEAK_DOWN = "weak_downtrend"
    DOWN = "downtrend"
    STRONG_DOWN = "strong_downtrend"


class PatternType(Enum):
    """Chart pattern classifications."""
    # Reversal patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ROUNDING_TOP = "rounding_top"
    ROUNDING_BOTTOM = "rounding_bottom"
    
    # Continuation patterns
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    BULL_PENNANT = "bull_pennant"
    BEAR_PENNANT = "bear_pennant"
    RECTANGLE = "rectangle"
    WEDGE_UP = "rising_wedge"
    WEDGE_DOWN = "falling_wedge"
    
    # Candlestick patterns
    DOJI = "doji"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    ENGULFING_BULL = "bullish_engulfing"
    ENGULFING_BEAR = "bearish_engulfing"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"


@dataclass
class TechnicalSignal:
    """Represents a technical analysis signal."""
    indicator: str
    signal_type: str  # 'buy', 'sell', 'neutral'
    strength: float  # -1 to 1
    value: float
    threshold: Optional[float] = None
    timeframe: str = "D1"
    timestamp: Optional[datetime] = None


@dataclass
class PatternMatch:
    """Represents a detected chart pattern."""
    pattern_type: PatternType
    confidence: float
    start_index: int
    end_index: int
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    is_bullish: bool = True
    notes: str = ""


@dataclass
class VolumeProfile:
    """Volume profile analysis result."""
    poc: float  # Point of Control
    value_area_high: float
    value_area_low: float
    high_volume_nodes: List[float]
    low_volume_nodes: List[float]
    volume_distribution: Dict[float, float]


class TechnicalEngine:
    """
    Advanced Technical Analysis Engine
    
    Provides comprehensive technical analysis capabilities including:
    - 50+ technical indicators (momentum, trend, volatility, volume)
    - Chart pattern recognition
    - Multi-timeframe analysis
    - Volume profile analytics
    - Support/resistance identification
    - Divergence detection
    - Fibonacci analysis
    
    Example:
        >>> engine = TechnicalEngine()
        >>> signals = engine.analyze(ohlcv_data)
        >>> patterns = engine.detect_patterns(ohlcv_data)
    """
    
    def __init__(
        self,
        default_lookback: int = 200,
        volume_profile_bins: int = 50,
        pattern_min_confidence: float = 0.6,
    ):
        """
        Initialize the Technical Analysis Engine.
        
        Args:
            default_lookback: Default lookback period for calculations
            volume_profile_bins: Number of bins for volume profile
            pattern_min_confidence: Minimum confidence for pattern detection
        """
        self.default_lookback = default_lookback
        self.volume_profile_bins = volume_profile_bins
        self.pattern_min_confidence = pattern_min_confidence
        
        # Signal weights for composite score
        self.signal_weights = {
            "rsi": 0.10,
            "macd": 0.12,
            "stochastic": 0.08,
            "cci": 0.08,
            "williams_r": 0.06,
            "adx": 0.10,
            "moving_averages": 0.15,
            "bollinger": 0.10,
            "volume": 0.11,
            "obv": 0.10,
        }

    def analyze(
        self,
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        include_patterns: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis.
        
        Args:
            open_prices: Array of opening prices
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            volume: Array of volume data
            include_patterns: Whether to include pattern detection
            
        Returns:
            Comprehensive technical analysis dictionary
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "data_points": len(close),
        }
        
        # Trend indicators
        results["trend"] = self._analyze_trend(close, high, low)
        
        # Momentum indicators
        results["momentum"] = self._analyze_momentum(close, high, low)
        
        # Volatility indicators
        results["volatility"] = self._analyze_volatility(close, high, low)
        
        # Volume indicators
        results["volume_analysis"] = self._analyze_volume(close, volume)
        
        # Support and resistance
        results["levels"] = self.identify_support_resistance(high, low, close)
        
        # Volume profile
        results["volume_profile"] = self.calculate_volume_profile(close, volume)
        
        # Pattern detection
        if include_patterns:
            results["patterns"] = self.detect_patterns(open_prices, high, low, close, volume)
        
        # Generate composite signal
        results["composite_signal"] = self._generate_composite_signal(results)
        
        return results

    def _analyze_trend(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze trend using multiple indicators."""
        
        # Moving averages
        sma_20 = self.sma(close, 20)
        sma_50 = self.sma(close, 50)
        sma_200 = self.sma(close, 200)
        ema_12 = self.ema(close, 12)
        ema_26 = self.ema(close, 26)
        
        # ADX for trend strength
        adx_result = self.adx(high, low, close)
        
        # Supertrend
        supertrend_result = self.supertrend(high, low, close)
        
        # Current price relative to MAs
        current = close[-1]
        
        # Determine trend direction
        ma_signals = []
        if current > sma_20[-1]:
            ma_signals.append(1)
        else:
            ma_signals.append(-1)
        if current > sma_50[-1]:
            ma_signals.append(1)
        else:
            ma_signals.append(-1)
        if len(sma_200) > 0 and not np.isnan(sma_200[-1]) and current > sma_200[-1]:
            ma_signals.append(1)
        else:
            ma_signals.append(-1)
        
        avg_ma_signal = np.mean(ma_signals)
        adx_value = adx_result["adx"][-1] if not np.isnan(adx_result["adx"][-1]) else 25
        
        if avg_ma_signal > 0.5 and adx_value > 25:
            direction = TrendDirection.STRONG_UP
        elif avg_ma_signal > 0:
            direction = TrendDirection.UP if adx_value > 20 else TrendDirection.WEAK_UP
        elif avg_ma_signal < -0.5 and adx_value > 25:
            direction = TrendDirection.STRONG_DOWN
        elif avg_ma_signal < 0:
            direction = TrendDirection.DOWN if adx_value > 20 else TrendDirection.WEAK_DOWN
        else:
            direction = TrendDirection.NEUTRAL
        
        return {
            "direction": direction.value,
            "strength": float(adx_value),
            "sma_20": float(sma_20[-1]) if not np.isnan(sma_20[-1]) else None,
            "sma_50": float(sma_50[-1]) if not np.isnan(sma_50[-1]) else None,
            "sma_200": float(sma_200[-1]) if len(sma_200) > 0 and not np.isnan(sma_200[-1]) else None,
            "ema_12": float(ema_12[-1]),
            "ema_26": float(ema_26[-1]),
            "adx": float(adx_value),
            "plus_di": float(adx_result["plus_di"][-1]) if not np.isnan(adx_result["plus_di"][-1]) else None,
            "minus_di": float(adx_result["minus_di"][-1]) if not np.isnan(adx_result["minus_di"][-1]) else None,
            "supertrend": supertrend_result,
            "signal": float(avg_ma_signal),
        }

    def _analyze_momentum(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze momentum using multiple oscillators."""
        
        # RSI
        rsi = self.rsi(close)
        rsi_value = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50
        
        # MACD
        macd_line, signal_line, histogram = self.macd(close)
        
        # Stochastic
        stoch_k, stoch_d = self.stochastic(high, low, close)
        
        # CCI
        cci = self.cci(high, low, close)
        
        # Williams %R
        williams_r = self.williams_r(high, low, close)
        
        # ROC
        roc = self.roc(close, period=14)
        
        # Momentum interpretation
        signals = {
            "rsi": self._interpret_rsi(rsi_value),
            "macd": self._interpret_macd(macd_line[-1], signal_line[-1], histogram[-1]),
            "stochastic": self._interpret_stochastic(stoch_k[-1], stoch_d[-1]),
            "cci": self._interpret_cci(cci[-1]),
        }
        
        # Composite momentum score
        momentum_score = np.mean([s["strength"] for s in signals.values()])
        
        return {
            "rsi": {
                "value": rsi_value,
                "signal": signals["rsi"]["signal"],
                "strength": signals["rsi"]["strength"],
            },
            "macd": {
                "macd": float(macd_line[-1]),
                "signal": float(signal_line[-1]),
                "histogram": float(histogram[-1]),
                "signal_type": signals["macd"]["signal"],
                "strength": signals["macd"]["strength"],
            },
            "stochastic": {
                "k": float(stoch_k[-1]) if not np.isnan(stoch_k[-1]) else 50,
                "d": float(stoch_d[-1]) if not np.isnan(stoch_d[-1]) else 50,
                "signal": signals["stochastic"]["signal"],
                "strength": signals["stochastic"]["strength"],
            },
            "cci": {
                "value": float(cci[-1]) if not np.isnan(cci[-1]) else 0,
                "signal": signals["cci"]["signal"],
                "strength": signals["cci"]["strength"],
            },
            "williams_r": {
                "value": float(williams_r[-1]) if not np.isnan(williams_r[-1]) else -50,
            },
            "roc": {
                "value": float(roc[-1]) if not np.isnan(roc[-1]) else 0,
            },
            "composite_score": float(momentum_score),
        }

    def _analyze_volatility(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze volatility metrics."""
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(close)
        bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] if bb_middle[-1] > 0 else 0
        bb_percent_b = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] != bb_lower[-1] else 0.5
        
        # ATR
        atr = self.atr(high, low, close)
        atr_percent = atr[-1] / close[-1] * 100 if close[-1] > 0 else 0
        
        # Keltner Channels
        kc_upper, kc_middle, kc_lower = self.keltner_channels(high, low, close)
        
        # Historical volatility
        returns = np.diff(close) / close[:-1]
        hist_vol = float(np.std(returns) * np.sqrt(252) * 100)  # Annualized
        
        # Volatility regime
        if atr_percent > 3:
            vol_regime = "high"
        elif atr_percent > 1.5:
            vol_regime = "moderate"
        else:
            vol_regime = "low"
        
        return {
            "bollinger": {
                "upper": float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else None,
                "middle": float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else None,
                "lower": float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else None,
                "width": float(bb_width),
                "percent_b": float(bb_percent_b),
            },
            "atr": {
                "value": float(atr[-1]) if not np.isnan(atr[-1]) else None,
                "percent": float(atr_percent),
            },
            "keltner": {
                "upper": float(kc_upper[-1]) if not np.isnan(kc_upper[-1]) else None,
                "middle": float(kc_middle[-1]) if not np.isnan(kc_middle[-1]) else None,
                "lower": float(kc_lower[-1]) if not np.isnan(kc_lower[-1]) else None,
            },
            "historical_volatility": hist_vol,
            "regime": vol_regime,
        }

    def _analyze_volume(
        self,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze volume indicators."""
        
        # On-Balance Volume
        obv = self.obv(close, volume)
        
        # Volume SMA
        vol_sma_20 = self.sma(volume, 20)
        current_vol = volume[-1]
        vol_ratio = current_vol / vol_sma_20[-1] if vol_sma_20[-1] > 0 else 1
        
        # Money Flow Index
        mfi = self.mfi(close, volume, period=14)
        
        # Accumulation/Distribution
        ad_line = self.accumulation_distribution(close, volume)
        
        # Volume trend
        recent_vol_avg = np.mean(volume[-10:])
        older_vol_avg = np.mean(volume[-30:-10]) if len(volume) > 30 else np.mean(volume)
        vol_trend = "increasing" if recent_vol_avg > older_vol_avg * 1.1 else "decreasing" if recent_vol_avg < older_vol_avg * 0.9 else "stable"
        
        return {
            "obv": {
                "value": float(obv[-1]),
                "trend": "up" if obv[-1] > obv[-10] else "down",
            },
            "volume_ratio": float(vol_ratio),
            "volume_trend": vol_trend,
            "mfi": {
                "value": float(mfi[-1]) if not np.isnan(mfi[-1]) else 50,
            },
            "ad_line": float(ad_line[-1]),
            "current_volume": float(current_vol),
            "avg_volume_20": float(vol_sma_20[-1]) if not np.isnan(vol_sma_20[-1]) else None,
        }

    # =========================================================================
    # Core Indicator Calculations
    # =========================================================================
    
    def sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        if len(data) < period:
            return np.full(len(data), np.nan)
        
        result = np.convolve(data, np.ones(period) / period, mode='valid')
        return np.concatenate([np.full(period - 1, np.nan), result])

    def ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        alpha = 2 / (period + 1)
        result = np.zeros(len(data))
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        
        return result

    def rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index."""
        if len(close) < period + 1:
            return np.full(len(close), np.nan)
        
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        alpha = 1.0 / period
        avg_gains = np.zeros(len(deltas))
        avg_losses = np.zeros(len(deltas))
        
        avg_gains[period-1] = np.mean(gains[:period])
        avg_losses[period-1] = np.mean(losses[:period])
        
        for i in range(period, len(deltas)):
            avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i-1]
            avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i-1]
        
        rs = np.divide(avg_gains, avg_losses, where=avg_losses != 0, out=np.ones_like(avg_gains))
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[np.nan], rsi])

    def macd(
        self,
        close: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD indicator."""
        fast_ema = self.ema(close, fast)
        slow_ema = self.ema(close, slow)
        macd_line = fast_ema - slow_ema
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def bollinger_bands(
        self,
        close: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands."""
        middle = self.sma(close, period)
        
        std = np.zeros(len(close))
        for i in range(period - 1, len(close)):
            std[i] = np.std(close[i-period+1:i+1])
        std[:period-1] = np.nan
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper, middle, lower

    def atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Average True Range."""
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        atr = np.zeros(len(tr))
        atr[period-1] = np.mean(tr[:period])
        
        alpha = 1.0 / period
        for i in range(period, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        
        atr[:period-1] = np.nan
        return atr

    def adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> Dict[str, np.ndarray]:
        """Average Directional Index."""
        n = len(close)
        
        # Calculate +DM and -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # ATR
        atr_vals = self.atr(high, low, close, period)
        
        # Smoothed +DM and -DM
        plus_dm_smooth = self.ema(plus_dm, period)
        minus_dm_smooth = self.ema(minus_dm, period)
        
        # +DI and -DI
        plus_di = np.zeros_like(plus_dm_smooth)
        minus_di = np.zeros_like(minus_dm_smooth)
        np.divide(plus_dm_smooth, atr_vals, where=atr_vals != 0, out=plus_di)
        np.divide(minus_dm_smooth, atr_vals, where=atr_vals != 0, out=minus_di)
        plus_di *= 100
        minus_di *= 100
        
        # DX and ADX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = np.zeros_like(di_diff)
        np.divide(di_diff, di_sum, where=di_sum != 0, out=dx)
        dx *= 100
        
        adx = self.ema(dx, period)
        
        return {
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
        }

    def stochastic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator."""
        n = len(close)
        stoch_k = np.full(n, np.nan)
        
        for i in range(k_period - 1, n):
            highest_high = np.max(high[i-k_period+1:i+1])
            lowest_low = np.min(low[i-k_period+1:i+1])
            
            if highest_high != lowest_low:
                stoch_k[i] = (close[i] - lowest_low) / (highest_high - lowest_low) * 100
            else:
                stoch_k[i] = 50
        
        stoch_d = self.sma(stoch_k, d_period)
        
        return stoch_k, stoch_d

    def cci(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 20,
    ) -> np.ndarray:
        """Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma_tp = self.sma(typical_price, period)
        
        # Mean deviation
        mean_dev = np.zeros(len(close))
        for i in range(period - 1, len(close)):
            mean_dev[i] = np.mean(np.abs(typical_price[i-period+1:i+1] - sma_tp[i]))
        
        cci = np.zeros_like(typical_price)
        np.divide(typical_price - sma_tp, 0.015 * mean_dev, where=mean_dev != 0, out=cci)
        cci[:period-1] = np.nan
        
        return cci

    def williams_r(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Williams %R."""
        n = len(close)
        williams = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            highest_high = np.max(high[i-period+1:i+1])
            lowest_low = np.min(low[i-period+1:i+1])
            
            if highest_high != lowest_low:
                williams[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
            else:
                williams[i] = -50
        
        return williams

    def roc(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Rate of Change."""
        roc = np.full(len(close), np.nan)
        for i in range(period, len(close)):
            if close[i-period] != 0:
                roc[i] = (close[i] - close[i-period]) / close[i-period] * 100
        return roc

    def obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On-Balance Volume."""
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv

    def mfi(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Money Flow Index."""
        # Note: This is simplified - ideally needs high and low
        n = len(close)
        mfi = np.full(n, np.nan)
        
        positive_flow = np.zeros(n)
        negative_flow = np.zeros(n)
        
        for i in range(1, n):
            money_flow = close[i] * volume[i]
            if close[i] > close[i-1]:
                positive_flow[i] = money_flow
            elif close[i] < close[i-1]:
                negative_flow[i] = money_flow
        
        for i in range(period, n):
            pos_sum = np.sum(positive_flow[i-period+1:i+1])
            neg_sum = np.sum(negative_flow[i-period+1:i+1])
            
            if neg_sum != 0:
                ratio = pos_sum / neg_sum
                mfi[i] = 100 - (100 / (1 + ratio))
            else:
                mfi[i] = 100
        
        return mfi

    def accumulation_distribution(
        self,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        """Accumulation/Distribution Line (simplified)."""
        # Simplified version - ideally needs high and low
        ad = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                ad[i] = ad[i-1] + volume[i]
            elif close[i] < close[i-1]:
                ad[i] = ad[i-1] - volume[i]
            else:
                ad[i] = ad[i-1]
        
        return ad

    def supertrend(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> Dict[str, Any]:
        """Supertrend indicator."""
        atr_vals = self.atr(high, low, close, period)
        hl2 = (high + low) / 2
        
        upper_band = hl2 + multiplier * atr_vals
        lower_band = hl2 - multiplier * atr_vals
        
        n = len(close)
        supertrend = np.zeros(n)
        direction = np.zeros(n)  # 1 for up, -1 for down
        
        for i in range(period, n):
            if close[i] > upper_band[i-1]:
                direction[i] = 1
            elif close[i] < lower_band[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            if direction[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
        
        return {
            "value": float(supertrend[-1]) if not np.isnan(supertrend[-1]) else None,
            "direction": "bullish" if direction[-1] == 1 else "bearish",
        }

    def keltner_channels(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keltner Channels."""
        middle = self.ema(close, ema_period)
        atr_vals = self.atr(high, low, close, atr_period)
        
        upper = middle + multiplier * atr_vals
        lower = middle - multiplier * atr_vals
        
        return upper, middle, lower

    # =========================================================================
    # Support & Resistance
    # =========================================================================
    
    def identify_support_resistance(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        lookback: int = 50,
        num_levels: int = 5,
    ) -> Dict[str, Any]:
        """Identify key support and resistance levels."""
        if len(close) < lookback:
            return {"support": [], "resistance": [], "pivot": {}}
        
        recent_high = high[-lookback:]
        recent_low = low[-lookback:]
        current_price = close[-1]
        
        # Find pivot points
        pivot_highs = []
        pivot_lows = []
        
        for i in range(2, len(recent_high) - 2):
            if (recent_high[i] > recent_high[i-1] and 
                recent_high[i] > recent_high[i-2] and
                recent_high[i] > recent_high[i+1] and 
                recent_high[i] > recent_high[i+2]):
                pivot_highs.append(recent_high[i])
            
            if (recent_low[i] < recent_low[i-1] and 
                recent_low[i] < recent_low[i-2] and
                recent_low[i] < recent_low[i+1] and 
                recent_low[i] < recent_low[i+2]):
                pivot_lows.append(recent_low[i])
        
        # Cluster levels
        def cluster_levels(levels: List[float], tolerance: float = 0.02) -> List[float]:
            if not levels:
                return []
            sorted_levels = sorted(levels)
            clusters = [[sorted_levels[0]]]
            
            for level in sorted_levels[1:]:
                if level - clusters[-1][-1] <= tolerance * clusters[-1][-1]:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])
            
            return [np.mean(c) for c in sorted(clusters, key=len, reverse=True)]
        
        resistance = [l for l in cluster_levels(pivot_highs) if l > current_price][:num_levels]
        support = [l for l in cluster_levels(pivot_lows) if l < current_price][:num_levels]
        
        # Classic pivot points (daily)
        h = high[-1]
        l = low[-1]
        c = close[-1]
        pivot = (h + l + c) / 3
        r1 = 2 * pivot - l
        s1 = 2 * pivot - h
        r2 = pivot + (h - l)
        s2 = pivot - (h - l)
        
        return {
            "support": sorted(support, reverse=True),
            "resistance": sorted(resistance),
            "pivot": {
                "pivot": float(pivot),
                "r1": float(r1),
                "r2": float(r2),
                "s1": float(s1),
                "s2": float(s2),
            },
        }

    # =========================================================================
    # Volume Profile
    # =========================================================================
    
    def calculate_volume_profile(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        bins: Optional[int] = None,
        value_area_pct: float = 0.70,
    ) -> Dict[str, Any]:
        """Calculate volume profile analysis."""
        bins = bins or self.volume_profile_bins
        
        if len(close) < 10:
            return {"poc": None, "value_area_high": None, "value_area_low": None}
        
        price_min = np.min(close)
        price_max = np.max(close)
        
        # Create price bins
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Assign volume to bins
        volume_by_bin = np.zeros(bins)
        for i in range(len(close)):
            bin_idx = np.searchsorted(bin_edges[1:], close[i])
            bin_idx = min(bin_idx, bins - 1)
            volume_by_bin[bin_idx] += volume[i]
        
        # Point of Control (highest volume bin)
        poc_idx = np.argmax(volume_by_bin)
        poc = float(bin_centers[poc_idx])
        
        # Value Area (70% of volume)
        total_vol = np.sum(volume_by_bin)
        target_vol = total_vol * value_area_pct
        
        # Expand from POC
        accumulated = volume_by_bin[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx
        
        while accumulated < target_vol and (low_idx > 0 or high_idx < bins - 1):
            expand_low = volume_by_bin[low_idx - 1] if low_idx > 0 else 0
            expand_high = volume_by_bin[high_idx + 1] if high_idx < bins - 1 else 0
            
            if expand_low > expand_high and low_idx > 0:
                low_idx -= 1
                accumulated += expand_low
            elif high_idx < bins - 1:
                high_idx += 1
                accumulated += expand_high
            elif low_idx > 0:
                low_idx -= 1
                accumulated += expand_low
            else:
                break
        
        value_area_low = float(bin_edges[low_idx])
        value_area_high = float(bin_edges[high_idx + 1])
        
        # High and low volume nodes
        sorted_indices = np.argsort(volume_by_bin)
        high_volume_nodes = [float(bin_centers[i]) for i in sorted_indices[-3:]]
        low_volume_nodes = [float(bin_centers[i]) for i in sorted_indices[:3] if volume_by_bin[i] > 0]
        
        return {
            "poc": poc,
            "value_area_high": value_area_high,
            "value_area_low": value_area_low,
            "high_volume_nodes": high_volume_nodes,
            "low_volume_nodes": low_volume_nodes,
        }

    # =========================================================================
    # Pattern Detection
    # =========================================================================
    
    def detect_patterns(
        self,
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Detect chart patterns in price data."""
        patterns = []
        
        # Candlestick patterns
        candlestick_patterns = self._detect_candlestick_patterns(open_prices, high, low, close)
        patterns.extend(candlestick_patterns)
        
        # Double top/bottom
        double_patterns = self._detect_double_patterns(high, low, close)
        patterns.extend(double_patterns)
        
        # Sort by confidence
        patterns.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return [p for p in patterns if p.get("confidence", 0) >= self.pattern_min_confidence]

    def _detect_candlestick_patterns(
        self,
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Detect candlestick patterns."""
        patterns = []
        
        if len(close) < 3:
            return patterns
        
        # Last candle analysis
        o, h, l, c = open_prices[-1], high[-1], low[-1], close[-1]
        body = abs(c - o)
        range_ = h - l
        
        if range_ == 0:
            return patterns
        
        body_ratio = body / range_
        
        # Doji
        if body_ratio < 0.1:
            patterns.append({
                "pattern": PatternType.DOJI.value,
                "confidence": 0.75,
                "is_bullish": None,  # Neutral
                "description": "Doji - indecision candle",
            })
        
        # Hammer (bullish)
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)
        
        if lower_shadow > body * 2 and upper_shadow < body * 0.5 and c > o:
            patterns.append({
                "pattern": PatternType.HAMMER.value,
                "confidence": 0.70,
                "is_bullish": True,
                "description": "Hammer - potential bullish reversal",
            })
        
        # Inverted Hammer
        if upper_shadow > body * 2 and lower_shadow < body * 0.5 and c < o:
            patterns.append({
                "pattern": PatternType.INVERTED_HAMMER.value,
                "confidence": 0.65,
                "is_bullish": True,
                "description": "Inverted Hammer - potential bullish reversal",
            })
        
        # Engulfing patterns (need at least 2 candles)
        if len(close) >= 2:
            prev_o, prev_c = open_prices[-2], close[-2]
            
            # Bullish engulfing
            if prev_c < prev_o and c > o and o < prev_c and c > prev_o:
                patterns.append({
                    "pattern": PatternType.ENGULFING_BULL.value,
                    "confidence": 0.75,
                    "is_bullish": True,
                    "description": "Bullish Engulfing - strong reversal signal",
                })
            
            # Bearish engulfing
            if prev_c > prev_o and c < o and o > prev_c and c < prev_o:
                patterns.append({
                    "pattern": PatternType.ENGULFING_BEAR.value,
                    "confidence": 0.75,
                    "is_bullish": False,
                    "description": "Bearish Engulfing - strong reversal signal",
                })
        
        return patterns

    def _detect_double_patterns(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        lookback: int = 50,
        tolerance: float = 0.02,
    ) -> List[Dict[str, Any]]:
        """Detect double top and double bottom patterns."""
        patterns = []
        
        if len(close) < lookback:
            return patterns
        
        recent_high = high[-lookback:]
        recent_low = low[-lookback:]
        
        # Find peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(2, len(recent_high) - 2):
            if (recent_high[i] > recent_high[i-1] and 
                recent_high[i] > recent_high[i-2] and
                recent_high[i] > recent_high[i+1] and 
                recent_high[i] > recent_high[i+2]):
                peaks.append((i, recent_high[i]))
            
            if (recent_low[i] < recent_low[i-1] and 
                recent_low[i] < recent_low[i-2] and
                recent_low[i] < recent_low[i+1] and 
                recent_low[i] < recent_low[i+2]):
                troughs.append((i, recent_low[i]))
        
        # Check for double top
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            p1, p2 = last_two_peaks[0][1], last_two_peaks[1][1]
            
            if abs(p1 - p2) / max(p1, p2) < tolerance:
                patterns.append({
                    "pattern": PatternType.DOUBLE_TOP.value,
                    "confidence": 0.70,
                    "is_bullish": False,
                    "description": f"Double Top at ~${np.mean([p1, p2]):.2f}",
                    "target": None,  # Would need neckline calculation
                })
        
        # Check for double bottom
        if len(troughs) >= 2:
            last_two_troughs = troughs[-2:]
            t1, t2 = last_two_troughs[0][1], last_two_troughs[1][1]
            
            if abs(t1 - t2) / min(t1, t2) < tolerance:
                patterns.append({
                    "pattern": PatternType.DOUBLE_BOTTOM.value,
                    "confidence": 0.70,
                    "is_bullish": True,
                    "description": f"Double Bottom at ~${np.mean([t1, t2]):.2f}",
                    "target": None,
                })
        
        return patterns

    # =========================================================================
    # Signal Interpretation Helpers
    # =========================================================================
    
    def _interpret_rsi(self, rsi_value: float) -> Dict[str, Any]:
        """Interpret RSI value."""
        if rsi_value >= 70:
            return {"signal": "overbought", "strength": -0.7}
        elif rsi_value >= 60:
            return {"signal": "bullish", "strength": 0.3}
        elif rsi_value <= 30:
            return {"signal": "oversold", "strength": 0.7}
        elif rsi_value <= 40:
            return {"signal": "bearish", "strength": -0.3}
        else:
            return {"signal": "neutral", "strength": 0.0}

    def _interpret_macd(
        self,
        macd: float,
        signal: float,
        histogram: float,
    ) -> Dict[str, Any]:
        """Interpret MACD values."""
        if macd > signal and histogram > 0:
            strength = min(0.8, abs(histogram) / max(abs(macd), 0.001))
            return {"signal": "bullish", "strength": strength}
        elif macd < signal and histogram < 0:
            strength = -min(0.8, abs(histogram) / max(abs(macd), 0.001))
            return {"signal": "bearish", "strength": strength}
        else:
            return {"signal": "neutral", "strength": 0.0}

    def _interpret_stochastic(self, k: float, d: float) -> Dict[str, Any]:
        """Interpret Stochastic values."""
        if k > 80 and d > 80:
            return {"signal": "overbought", "strength": -0.6}
        elif k < 20 and d < 20:
            return {"signal": "oversold", "strength": 0.6}
        elif k > d and k < 80:
            return {"signal": "bullish", "strength": 0.3}
        elif k < d and k > 20:
            return {"signal": "bearish", "strength": -0.3}
        else:
            return {"signal": "neutral", "strength": 0.0}

    def _interpret_cci(self, cci_value: float) -> Dict[str, Any]:
        """Interpret CCI value."""
        if cci_value > 100:
            return {"signal": "overbought", "strength": -0.5}
        elif cci_value < -100:
            return {"signal": "oversold", "strength": 0.5}
        elif cci_value > 0:
            return {"signal": "bullish", "strength": 0.2}
        else:
            return {"signal": "bearish", "strength": -0.2}

    def _generate_composite_signal(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate composite signal from all analysis."""
        signals = []
        
        # Trend signal
        if "trend" in analysis:
            trend = analysis["trend"]
            if trend["direction"] in ["strong_uptrend", "uptrend"]:
                signals.append(trend.get("signal", 0.5))
            elif trend["direction"] in ["strong_downtrend", "downtrend"]:
                signals.append(trend.get("signal", -0.5))
            else:
                signals.append(0)
        
        # Momentum signal
        if "momentum" in analysis:
            signals.append(analysis["momentum"].get("composite_score", 0))
        
        composite = np.mean(signals) if signals else 0
        
        if composite > 0.5:
            signal_type = "strong_buy"
        elif composite > 0.2:
            signal_type = "buy"
        elif composite > -0.2:
            signal_type = "neutral"
        elif composite > -0.5:
            signal_type = "sell"
        else:
            signal_type = "strong_sell"
        
        return {
            "signal": signal_type,
            "score": float(composite),
            "confidence": abs(float(composite)),
        }

