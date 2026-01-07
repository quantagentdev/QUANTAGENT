"""
QuantTraderAgent - Core Implementation

The main agent class that orchestrates all quantitative analysis modules
and provides comprehensive market intelligence through the Swarms framework.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import numpy as np
from datetime import datetime, timedelta
import threading
import json

try:
    from swarms.structs.agent import Agent
except ImportError:
    raise ImportError(
        "swarms is required for QuantTraderAgent. Install with: pip install swarms"
    )

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Market condition classifications."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class TimeFrame(Enum):
    """Supported analysis timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


class SignalStrength(Enum):
    """Signal strength classifications."""
    STRONG_BUY = 5
    BUY = 4
    WEAK_BUY = 3
    NEUTRAL = 2
    WEAK_SELL = 1
    SELL = 0
    STRONG_SELL = -1


@dataclass
class PriceTarget:
    """Represents a price target with metadata."""
    price: float
    timeframe: str
    confidence: float
    basis: str  # Technical/Fundamental/Hybrid
    support_level: float
    resistance_level: float
    risk_reward_ratio: float
    invalidation_price: float
    probability: float
    reasoning: str


@dataclass
class AnalysisResult:
    """Comprehensive analysis result structure."""
    symbol: str
    timestamp: datetime
    current_price: float
    market_condition: MarketCondition
    trend_direction: str
    trend_strength: float
    momentum_score: float
    volatility_percentile: float
    volume_profile: Dict[str, Any]
    support_levels: List[float]
    resistance_levels: List[float]
    price_targets: List[PriceTarget]
    risk_metrics: Dict[str, float]
    signal_strength: SignalStrength
    confidence_score: float
    key_observations: List[str]
    risk_warnings: List[str]


@dataclass
class SecurityProfile:
    """Profile of a security being analyzed."""
    symbol: str
    asset_class: str  # stock, crypto, forex, commodity, index
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    avg_volume: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    pe_ratio: Optional[float] = None
    historical_volatility: Optional[float] = None
    correlation_to_market: Optional[float] = None


class QuantTraderAgent(Agent):
    """
    QuantTraderAgent - Elite Quantitative Trading Analysis Agent
    
    A sophisticated AI agent that synthesizes technical indicators, fundamental 
    valuations, market sentiment, and macroeconomic trends into actionable 
    intelligence with precise price targets and risk parameters.
    
    Built on the Swarms framework for enterprise-grade agent orchestration.
    
    Key Capabilities:
    - Multi-dimensional technical analysis across timeframes
    - Fundamental valuation using DCF, multiples, and regression
    - Market regime detection and adaptive strategy selection
    - AI-powered price target generation with confidence intervals
    - Comprehensive risk management (VaR, CVaR, position sizing)
    - Real-time signal aggregation and confluence detection
    
    Example:
        >>> from quantagent import QuantTraderAgent
        >>> agent = QuantTraderAgent(model_name="gpt-4o")
        >>> analysis = agent.analyze_security(
        ...     symbol="AAPL",
        ...     price_data=price_df,
        ...     fundamentals=fundamentals_dict
        ... )
        >>> print(analysis.price_targets)
    """
    
    # Default system prompt for the Quant Trader Agent
    QUANT_TRADER_SYSTEM_PROMPT = """You are Quant Trader Agent, a highly sophisticated financial analyst AI with the combined capabilities of a Wall Street veteran, a quantitative mathematician, and a behavioral economist. Your core competency lies in synthesizing vast amounts of financial data—from technical indicators and fundamental metrics to macroeconomic trends and market sentiment—into actionable intelligence.

## Your Analytical Framework

When analyzing any security, you operate with the rigor of a professional trader combined with the analytical depth of a quantitative researcher. Your analysis always begins by gathering comprehensive data across multiple dimensions: price action and volume patterns, fundamental valuations, sector dynamics, macroeconomic context, and market microstructure. You understand that markets are complex adaptive systems where multiple factors interact simultaneously, and you never rely on single indicators or simplistic narratives.

Your technical analysis incorporates classical chart patterns, statistical measures of momentum and mean reversion, volume profile analysis, options flow data when available, and multi-timeframe confirmation. You recognize support and resistance levels not just as lines on a chart but as zones where order flow dynamics shift. When you identify patterns, you always contextualize them within the broader market structure and acknowledge their probabilistic nature rather than treating them as certainties.

On the fundamental side, you dissect financial statements with forensic precision, analyzing not just headline metrics but cash flow quality, accounting choices, competitive positioning, management effectiveness, and industry-specific key performance indicators. You understand that a company's intrinsic value emerges from its ability to generate sustainable cash flows, but you also recognize that market prices often diverge from intrinsic value for extended periods based on sentiment, momentum, and structural factors.

## Your Communication Style

When presenting your analysis, you structure your insights in clear, flowing prose that builds logically from observation to interpretation to implication. You begin by establishing the current state of the security—where it stands in terms of price, trend, and market context. You then layer in your multi-dimensional analysis, weaving together technical signals, fundamental considerations, and market dynamics into a coherent narrative.

You are precise with numbers and specific about timeframes, price levels, and probabilities. When you identify key levels or targets, you explain the reasoning behind them—whether they're derived from Fibonacci retracements, prior volume nodes, fundamental valuation models, or options market positioning. You quantify whenever possible, providing percentage changes, dollar amounts, and time horizons to make your analysis actionable.

Your tone is confident but intellectually honest. You acknowledge uncertainty where it exists and clearly distinguish between high-conviction views based on strong evidence and more speculative scenarios. You never present opinions as facts, but you also don't hedge so excessively that your analysis becomes meaningless.

## Delivering Stock Analysis

When analyzing a stock's current performance, you paint a comprehensive picture that captures both its recent trajectory and its positioning within the broader market landscape. You describe the prevailing trend—whether bullish, bearish, or range-bound—and quantify the magnitude of recent moves. You contextualize current performance against relevant benchmarks, sector peers, and historical volatility patterns. You identify the key drivers behind recent price action.

You assess momentum using multiple lenses: raw price momentum, relative strength versus indices and sectors, volume characteristics, and derivative indicators like RSI or MACD when relevant. You evaluate whether the current move appears sustainable based on participation, follow-through, and alignment with fundamental developments.

Your assessment of support and resistance incorporates technical levels from multiple timeframes, psychological round numbers, prior consolidation zones, moving averages that have historically served as dynamic support or resistance, and volume profile data showing where significant trading activity has occurred.

## Setting Price Targets

You establish three intermediate price targets that represent progressively more ambitious scenarios, each grounded in specific technical or fundamental logic. These aren't arbitrary numbers but levels where you expect meaningful reactions based on the interplay of technical structure, fundamental valuation, and market psychology.

Your first target represents a near-term objective achievable within days to a few weeks. Your second target extends further, assuming trend continuation over several weeks to a couple of months. Your third target represents the security's potential under favorable conditions over a quarterly timeframe or longer.

For each target, you specify the exact price level, approximate timeframe, key assumptions, and what would invalidate the thesis.

## Critical Guidelines

You never make guarantees about future price movements because markets are probabilistic systems. You present scenarios and probabilities, not certainties. You distinguish between analytical assessment and investment advice - you provide information and analysis but defer to individuals to make their own decisions.

Your ultimate goal is to deliver analysis that helps traders and investors make more informed decisions by providing clarity about current positioning, potential pathways, and associated risks."""

    def __init__(
        self,
        agent_name: str = "quant-trader-agent",
        agent_description: str = "Elite Quantitative Trading Analysis Agent",
        model_name: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        max_loops: int = 3,
        risk_free_rate: float = 0.05,
        confidence_level: float = 0.95,
        default_timeframe: TimeFrame = TimeFrame.D1,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 300,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the QuantTraderAgent.
        
        Args:
            agent_name: Name identifier for the agent
            agent_description: Description of agent capabilities
            model_name: LLM model to use (default: gpt-4o)
            system_prompt: Custom system prompt (uses default if None)
            max_loops: Maximum reasoning loops for complex analysis
            risk_free_rate: Risk-free rate for Sharpe calculations (default: 5%)
            confidence_level: Confidence level for VaR calculations (default: 95%)
            default_timeframe: Default analysis timeframe
            enable_caching: Whether to cache analysis results
            cache_ttl_seconds: Cache time-to-live in seconds
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to parent Agent
        """
        
        # Get the analysis schema for structured output
        quant_schema = self._get_quant_analysis_schema()
        
        # Initialize parent Agent class
        super().__init__(
            agent_name=agent_name,
            agent_description=agent_description,
            model_name=model_name,
            system_prompt=system_prompt or self.QUANT_TRADER_SYSTEM_PROMPT,
            max_loops=1,  # We handle multi-loop reasoning internally
            tools_list_dictionary=[quant_schema],
            output_type="final",
            **kwargs,
        )
        
        # Configuration
        self.quant_max_loops = max_loops
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.default_timeframe = default_timeframe
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self.seed = seed if seed is not None else 42
        self._rng = np.random.default_rng(self.seed)
        
        # Internal state
        self._analysis_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_lock = threading.Lock()
        self._analysis_history: List[Dict[str, Any]] = []
        
        # Technical analysis configuration
        self.indicator_weights: Dict[str, float] = {
            "trend": 0.25,
            "momentum": 0.20,
            "volatility": 0.15,
            "volume": 0.15,
            "pattern": 0.15,
            "sentiment": 0.10,
        }
        
        # Price level tracking
        self.support_resistance_memory: Dict[str, List[Tuple[float, float, str]]] = {}
        
        # Market regime state
        self.current_regime: Optional[MarketCondition] = None
        self.regime_confidence: float = 0.0
        
        logger.info(f"QuantTraderAgent initialized: {agent_name}")

    @staticmethod
    def _get_quant_analysis_schema() -> Dict[str, Any]:
        """Returns the JSON schema for structured quantitative analysis output."""
        return {
            "type": "function",
            "function": {
                "name": "generate_quantitative_analysis",
                "description": "Generates comprehensive quantitative trading analysis with price targets",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "market_overview": {
                            "type": "string",
                            "description": "Current market conditions and context"
                        },
                        "trend_analysis": {
                            "type": "object",
                            "properties": {
                                "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                                "strength": {"type": "number", "minimum": 0, "maximum": 100},
                                "timeframe": {"type": "string"},
                                "reasoning": {"type": "string"}
                            }
                        },
                        "technical_signals": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "indicator": {"type": "string"},
                                    "signal": {"type": "string"},
                                    "value": {"type": "number"},
                                    "interpretation": {"type": "string"}
                                }
                            }
                        },
                        "support_resistance": {
                            "type": "object",
                            "properties": {
                                "support_levels": {"type": "array", "items": {"type": "number"}},
                                "resistance_levels": {"type": "array", "items": {"type": "number"}},
                                "key_level_reasoning": {"type": "string"}
                            }
                        },
                        "price_targets": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "target_price": {"type": "number"},
                                    "timeframe": {"type": "string"},
                                    "probability": {"type": "number"},
                                    "basis": {"type": "string"},
                                    "invalidation_level": {"type": "number"},
                                    "reasoning": {"type": "string"}
                                }
                            },
                            "description": "Three price targets: near-term, intermediate, and extended"
                        },
                        "risk_assessment": {
                            "type": "object",
                            "properties": {
                                "risk_reward_ratio": {"type": "number"},
                                "max_drawdown_estimate": {"type": "number"},
                                "volatility_assessment": {"type": "string"},
                                "key_risks": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "momentum_analysis": {
                            "type": "string",
                            "description": "Assessment of current momentum and sustainability"
                        },
                        "fundamental_context": {
                            "type": "string",
                            "description": "Relevant fundamental factors affecting the security"
                        },
                        "actionable_summary": {
                            "type": "string",
                            "description": "Concise actionable intelligence summary"
                        },
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Overall confidence in the analysis (0-100)"
                        }
                    },
                    "required": [
                        "market_overview",
                        "trend_analysis",
                        "technical_signals",
                        "support_resistance",
                        "price_targets",
                        "risk_assessment",
                        "momentum_analysis",
                        "actionable_summary",
                        "confidence_score"
                    ]
                }
            }
        }

    def _build_analysis_prompt(
        self,
        symbol: str,
        price_data: Optional[Dict[str, Any]] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
        market_context: Optional[str] = None,
    ) -> str:
        """Build the analysis prompt with all available data."""
        prompt_parts = [
            f"Perform a comprehensive quantitative analysis of {symbol}.",
            "",
        ]
        
        if price_data:
            prompt_parts.append("## Price Data Summary")
            prompt_parts.append(f"Current Price: ${price_data.get('current_price', 'N/A')}")
            prompt_parts.append(f"52-Week High: ${price_data.get('high_52w', 'N/A')}")
            prompt_parts.append(f"52-Week Low: ${price_data.get('low_52w', 'N/A')}")
            if 'returns' in price_data:
                prompt_parts.append(f"Returns: {price_data['returns']}")
            if 'volume' in price_data:
                prompt_parts.append(f"Volume Profile: {price_data['volume']}")
            prompt_parts.append("")
        
        if fundamentals:
            prompt_parts.append("## Fundamental Data")
            for key, value in fundamentals.items():
                prompt_parts.append(f"{key}: {value}")
            prompt_parts.append("")
        
        if market_context:
            prompt_parts.append("## Market Context")
            prompt_parts.append(market_context)
            prompt_parts.append("")
        
        prompt_parts.append("Provide your analysis following the structured format.")
        
        return "\n".join(prompt_parts)

    def analyze_security(
        self,
        symbol: str,
        price_data: Optional[Dict[str, Any]] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
        market_context: Optional[str] = None,
        timeframe: Optional[TimeFrame] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quantitative analysis on a security.
        
        Args:
            symbol: Security symbol/ticker
            price_data: Historical price data and derived metrics
            fundamentals: Fundamental data (financials, ratios, etc.)
            market_context: Additional market context or news
            timeframe: Analysis timeframe (default: D1)
            use_cache: Whether to use cached results if available
            
        Returns:
            Comprehensive analysis result dictionary
        """
        # Check cache
        cache_key = f"{symbol}_{timeframe or self.default_timeframe}"
        if use_cache and self.enable_caching:
            cached = self._get_cached_analysis(cache_key)
            if cached:
                logger.debug(f"Returning cached analysis for {symbol}")
                return cached
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(
            symbol=symbol,
            price_data=price_data,
            fundamentals=fundamentals,
            market_context=market_context,
        )
        
        # Run multi-loop analysis
        analysis_steps = []
        current_prompt = prompt
        
        for i in range(self.quant_max_loops):
            step_result = super().run(current_prompt)
            analysis_steps.append({
                "step": i + 1,
                "analysis": step_result,
                "timestamp": datetime.now().isoformat(),
            })
            
            # Refine prompt for next iteration
            if i < self.quant_max_loops - 1:
                current_prompt = f"{prompt}\n\nPrevious Analysis:\n{step_result}\n\nRefine and deepen the analysis."
        
        # Synthesize final analysis
        final_analysis = self._synthesize_analysis(symbol, analysis_steps)
        
        # Cache result
        if self.enable_caching:
            self._cache_analysis(cache_key, final_analysis)
        
        # Store in history
        self._analysis_history.append({
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "result": final_analysis,
        })
        
        return final_analysis

    def _synthesize_analysis(
        self,
        symbol: str,
        analysis_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Synthesize analysis steps into final result."""
        synthesis_prompt = f"""Based on the multi-step analysis performed for {symbol}, 
        synthesize a final comprehensive trading analysis report.
        
        Analysis Steps:
        {json.dumps(analysis_steps, indent=2, default=str)}
        
        Provide the final synthesized analysis."""
        
        final_response = super().run(synthesis_prompt)
        
        return {
            "symbol": symbol,
            "analysis": final_response,
            "steps": analysis_steps,
            "timestamp": datetime.now().isoformat(),
            "agent_version": "1.0.0",
        }

    def _get_cached_analysis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis if valid."""
        with self._cache_lock:
            if cache_key in self._analysis_cache:
                result, timestamp = self._analysis_cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl_seconds):
                    return result
                else:
                    del self._analysis_cache[cache_key]
        return None

    def _cache_analysis(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache analysis result."""
        with self._cache_lock:
            self._analysis_cache[cache_key] = (result, datetime.now())

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        with self._cache_lock:
            self._analysis_cache.clear()

    # =========================================================================
    # Technical Analysis Methods
    # =========================================================================
    
    def calculate_rsi(
        self,
        prices: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Array of closing prices
            period: RSI period (default: 14)
            
        Returns:
            Array of RSI values
        """
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses using EMA
        alpha = 1.0 / period
        avg_gains = np.zeros(len(deltas))
        avg_losses = np.zeros(len(deltas))
        
        avg_gains[period-1] = np.mean(gains[:period])
        avg_losses[period-1] = np.mean(losses[:period])
        
        for i in range(period, len(deltas)):
            avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i-1]
            avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i-1]
        
        rs = np.divide(avg_gains, avg_losses, where=avg_losses != 0)
        rsi = 100 - (100 / (1 + rs))
        
        # Prepend NaN for the first value (since we took diff)
        return np.concatenate([[np.nan], rsi])

    def calculate_macd(
        self,
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Array of closing prices
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        def ema(data: np.ndarray, period: int) -> np.ndarray:
            alpha = 2 / (period + 1)
            result = np.zeros(len(data))
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        fast_ema = ema(prices, fast_period)
        slow_ema = ema(prices, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(
        self,
        prices: np.ndarray,
        period: int = 20,
        num_std: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Array of closing prices
            period: Moving average period (default: 20)
            num_std: Number of standard deviations (default: 2.0)
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle = np.convolve(prices, np.ones(period)/period, mode='valid')
        
        # Pad the beginning with NaN
        pad_length = len(prices) - len(middle)
        middle = np.concatenate([np.full(pad_length, np.nan), middle])
        
        # Calculate rolling std
        std = np.zeros(len(prices))
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i-period+1:i+1])
        std[:period-1] = np.nan
        
        upper = middle + num_std * std
        lower = middle - num_std * std
        
        return upper, middle, lower

    def calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            period: ATR period (default: 14)
            
        Returns:
            Array of ATR values
        """
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Calculate ATR using EMA
        atr = np.zeros(len(tr))
        atr[period-1] = np.mean(tr[:period])
        
        alpha = 1.0 / period
        for i in range(period, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        
        atr[:period-1] = np.nan
        return atr

    def calculate_vwap(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            volume: Array of volume data
            
        Returns:
            Array of VWAP values
        """
        typical_price = (high + low + close) / 3
        cumulative_tp_volume = np.cumsum(typical_price * volume)
        cumulative_volume = np.cumsum(volume)
        
        vwap = np.divide(
            cumulative_tp_volume,
            cumulative_volume,
            where=cumulative_volume != 0
        )
        
        return vwap

    def detect_divergence(
        self,
        prices: np.ndarray,
        indicator: np.ndarray,
        lookback: int = 20,
    ) -> Dict[str, Any]:
        """
        Detect bullish and bearish divergences between price and indicator.
        
        Args:
            prices: Array of prices
            indicator: Array of indicator values (e.g., RSI)
            lookback: Lookback period for divergence detection
            
        Returns:
            Dictionary with divergence signals
        """
        if len(prices) < lookback or len(indicator) < lookback:
            return {"bullish": False, "bearish": False, "confidence": 0.0}
        
        recent_prices = prices[-lookback:]
        recent_indicator = indicator[-lookback:]
        
        # Find local extrema
        price_highs = []
        price_lows = []
        ind_highs = []
        ind_lows = []
        
        for i in range(1, len(recent_prices) - 1):
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                price_highs.append((i, recent_prices[i]))
            if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                price_lows.append((i, recent_prices[i]))
            if not np.isnan(recent_indicator[i]):
                if recent_indicator[i] > recent_indicator[i-1] and recent_indicator[i] > recent_indicator[i+1]:
                    ind_highs.append((i, recent_indicator[i]))
                if recent_indicator[i] < recent_indicator[i-1] and recent_indicator[i] < recent_indicator[i+1]:
                    ind_lows.append((i, recent_indicator[i]))
        
        bullish_div = False
        bearish_div = False
        confidence = 0.0
        
        # Check for bearish divergence (higher high in price, lower high in indicator)
        if len(price_highs) >= 2 and len(ind_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and ind_highs[-1][1] < ind_highs[-2][1]:
                bearish_div = True
                confidence = 0.7
        
        # Check for bullish divergence (lower low in price, higher low in indicator)
        if len(price_lows) >= 2 and len(ind_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and ind_lows[-1][1] > ind_lows[-2][1]:
                bullish_div = True
                confidence = 0.7
        
        return {
            "bullish": bullish_div,
            "bearish": bearish_div,
            "confidence": confidence,
        }

    def identify_support_resistance(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        lookback: int = 50,
        num_levels: int = 5,
        tolerance: float = 0.02,
    ) -> Dict[str, List[float]]:
        """
        Identify key support and resistance levels using pivot points and clustering.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            lookback: Lookback period for level identification
            num_levels: Maximum number of levels to return
            tolerance: Clustering tolerance as percentage
            
        Returns:
            Dictionary with support and resistance levels
        """
        if len(close) < lookback:
            return {"support": [], "resistance": []}
        
        recent_high = high[-lookback:]
        recent_low = low[-lookback:]
        current_price = close[-1]
        
        # Find pivot highs and lows
        pivot_highs = []
        pivot_lows = []
        
        for i in range(2, len(recent_high) - 2):
            # Pivot high
            if (recent_high[i] > recent_high[i-1] and 
                recent_high[i] > recent_high[i-2] and
                recent_high[i] > recent_high[i+1] and 
                recent_high[i] > recent_high[i+2]):
                pivot_highs.append(recent_high[i])
            
            # Pivot low
            if (recent_low[i] < recent_low[i-1] and 
                recent_low[i] < recent_low[i-2] and
                recent_low[i] < recent_low[i+1] and 
                recent_low[i] < recent_low[i+2]):
                pivot_lows.append(recent_low[i])
        
        # Cluster nearby levels
        def cluster_levels(levels: List[float], tol: float) -> List[float]:
            if not levels:
                return []
            
            sorted_levels = sorted(levels)
            clusters = [[sorted_levels[0]]]
            
            for level in sorted_levels[1:]:
                if level - clusters[-1][-1] <= tol * clusters[-1][-1]:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])
            
            # Return cluster centers weighted by cluster size
            return [np.mean(c) for c in sorted(clusters, key=len, reverse=True)[:num_levels]]
        
        resistance_levels = [l for l in cluster_levels(pivot_highs, tolerance) if l > current_price]
        support_levels = [l for l in cluster_levels(pivot_lows, tolerance) if l < current_price]
        
        return {
            "support": sorted(support_levels, reverse=True)[:num_levels],
            "resistance": sorted(resistance_levels)[:num_levels],
        }

    # =========================================================================
    # Risk Management Methods
    # =========================================================================
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: Optional[float] = None,
        method: str = "historical",
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (default: instance setting)
            method: Calculation method ('historical', 'parametric', 'cornish_fisher')
            
        Returns:
            VaR as a positive number (potential loss)
        """
        confidence_level = confidence_level or self.confidence_level
        
        if len(returns) < 30:
            logger.warning("Insufficient data for reliable VaR calculation")
            return np.nan
        
        if method == "historical":
            var = -np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == "parametric":
            from scipy.stats import norm
            mu = np.mean(returns)
            sigma = np.std(returns)
            var = -(mu + sigma * norm.ppf(1 - confidence_level))
        
        elif method == "cornish_fisher":
            # Cornish-Fisher expansion for non-normal distributions
            from scipy.stats import norm
            mu = np.mean(returns)
            sigma = np.std(returns)
            skew = ((returns - mu) ** 3).mean() / sigma ** 3
            kurt = ((returns - mu) ** 4).mean() / sigma ** 4 - 3
            
            z = norm.ppf(1 - confidence_level)
            z_cf = (z + (z**2 - 1) * skew / 6 + 
                    (z**3 - 3*z) * kurt / 24 - 
                    (2*z**3 - 5*z) * skew**2 / 36)
            
            var = -(mu + sigma * z_cf)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return float(var)

    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence_level: Optional[float] = None,
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (default: instance setting)
            
        Returns:
            CVaR as a positive number (expected loss beyond VaR)
        """
        confidence_level = confidence_level or self.confidence_level
        
        var = self.calculate_var(returns, confidence_level, method="historical")
        if np.isnan(var):
            return np.nan
        
        # CVaR is the expected loss given that loss exceeds VaR
        tail_returns = returns[returns <= -var]
        if len(tail_returns) == 0:
            return var
        
        cvar = -np.mean(tail_returns)
        return float(cvar)

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[float] = None,
        annualization_factor: int = 252,
    ) -> float:
        """
        Calculate Sharpe Ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (default: instance setting)
            annualization_factor: Trading days per year (default: 252)
            
        Returns:
            Annualized Sharpe Ratio
        """
        risk_free_rate = risk_free_rate or self.risk_free_rate
        
        if len(returns) < 2:
            return np.nan
        
        excess_returns = returns - risk_free_rate / annualization_factor
        
        if np.std(excess_returns) == 0:
            return np.nan
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return float(sharpe * np.sqrt(annualization_factor))

    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[float] = None,
        annualization_factor: int = 252,
    ) -> float:
        """
        Calculate Sortino Ratio (downside risk-adjusted return).
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (default: instance setting)
            annualization_factor: Trading days per year (default: 252)
            
        Returns:
            Annualized Sortino Ratio
        """
        risk_free_rate = risk_free_rate or self.risk_free_rate
        
        if len(returns) < 2:
            return np.nan
        
        excess_returns = returns - risk_free_rate / annualization_factor
        downside_returns = np.minimum(excess_returns, 0)
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return np.nan
        
        sortino = np.mean(excess_returns) / downside_std
        return float(sortino * np.sqrt(annualization_factor))

    def calculate_max_drawdown(
        self,
        prices: np.ndarray,
    ) -> Tuple[float, int, int]:
        """
        Calculate Maximum Drawdown and its duration.
        
        Args:
            prices: Array of prices
            
        Returns:
            Tuple of (max_drawdown, peak_index, trough_index)
        """
        if len(prices) < 2:
            return (0.0, 0, 0)
        
        cumulative_max = np.maximum.accumulate(prices)
        drawdowns = (prices - cumulative_max) / cumulative_max
        
        max_dd = float(np.min(drawdowns))
        trough_idx = int(np.argmin(drawdowns))
        peak_idx = int(np.argmax(prices[:trough_idx+1]))
        
        return (max_dd, peak_idx, trough_idx)

    def calculate_optimal_position_size(
        self,
        account_value: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade: float = 0.02,
    ) -> Dict[str, float]:
        """
        Calculate optimal position size using fixed fractional risk.
        
        Args:
            account_value: Total account value
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            risk_per_trade: Maximum risk per trade as fraction (default: 2%)
            
        Returns:
            Dictionary with position sizing information
        """
        risk_amount = account_value * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return {"error": "Stop loss cannot equal entry price"}
        
        shares = risk_amount / risk_per_share
        position_value = shares * entry_price
        
        return {
            "shares": float(shares),
            "position_value": float(position_value),
            "risk_amount": float(risk_amount),
            "risk_per_share": float(risk_per_share),
            "position_pct_of_portfolio": float(position_value / account_value * 100),
        }

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25,
    ) -> float:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            kelly_fraction: Fraction of full Kelly to use (default: 0.25)
            
        Returns:
            Optimal position size as fraction of portfolio
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly formula: f* = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - p
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply fractional Kelly for safety
        fractional_kelly = kelly * kelly_fraction
        
        return float(max(0, min(1, fractional_kelly)))

    # =========================================================================
    # Price Target Methods
    # =========================================================================
    
    def generate_price_targets(
        self,
        current_price: float,
        support_levels: List[float],
        resistance_levels: List[float],
        atr: float,
        trend_direction: str,
        fundamental_fair_value: Optional[float] = None,
    ) -> List[PriceTarget]:
        """
        Generate three price targets based on technical and fundamental analysis.
        
        Args:
            current_price: Current market price
            support_levels: List of support levels
            resistance_levels: List of resistance levels
            atr: Average True Range
            trend_direction: 'bullish', 'bearish', or 'neutral'
            fundamental_fair_value: Fundamental fair value estimate if available
            
        Returns:
            List of three PriceTarget objects
        """
        targets = []
        
        if trend_direction == "bullish":
            # Near-term target: First resistance or 1.5 ATR
            if resistance_levels:
                target1_price = resistance_levels[0]
            else:
                target1_price = current_price + 1.5 * atr
            
            # Intermediate target: Second resistance or 3 ATR
            if len(resistance_levels) > 1:
                target2_price = resistance_levels[1]
            else:
                target2_price = current_price + 3 * atr
            
            # Extended target: Based on fundamental value or 5 ATR
            if fundamental_fair_value and fundamental_fair_value > current_price:
                target3_price = fundamental_fair_value
            else:
                target3_price = current_price + 5 * atr
            
            stop_loss = support_levels[0] if support_levels else current_price - 2 * atr
            
        elif trend_direction == "bearish":
            # Targets are downside
            if support_levels:
                target1_price = support_levels[0]
            else:
                target1_price = current_price - 1.5 * atr
            
            if len(support_levels) > 1:
                target2_price = support_levels[1]
            else:
                target2_price = current_price - 3 * atr
            
            if fundamental_fair_value and fundamental_fair_value < current_price:
                target3_price = fundamental_fair_value
            else:
                target3_price = current_price - 5 * atr
            
            stop_loss = resistance_levels[0] if resistance_levels else current_price + 2 * atr
            
        else:  # Neutral
            # Range-bound targets
            target1_price = current_price + atr if resistance_levels else current_price + atr
            target2_price = resistance_levels[0] if resistance_levels else current_price + 2 * atr
            target3_price = resistance_levels[1] if len(resistance_levels) > 1 else current_price + 3 * atr
            stop_loss = support_levels[0] if support_levels else current_price - 1.5 * atr
        
        # Create targets with metadata
        target_configs = [
            ("Near-term", target1_price, "1-2 weeks", 0.65, "Technical - First target level"),
            ("Intermediate", target2_price, "4-8 weeks", 0.45, "Technical - Extended move"),
            ("Extended", target3_price, "3+ months", 0.30, "Technical/Fundamental blend"),
        ]
        
        for name, price, timeframe, prob, basis in target_configs:
            rr_ratio = abs(price - current_price) / abs(current_price - stop_loss) if stop_loss != current_price else 0
            
            targets.append(PriceTarget(
                price=round(price, 2),
                timeframe=timeframe,
                confidence=prob,
                basis=basis,
                support_level=support_levels[0] if support_levels else current_price - atr,
                resistance_level=resistance_levels[0] if resistance_levels else current_price + atr,
                risk_reward_ratio=round(rr_ratio, 2),
                invalidation_price=round(stop_loss, 2),
                probability=prob,
                reasoning=f"{name} target at ${price:.2f} based on {basis}",
            ))
        
        return targets

    # =========================================================================
    # Market Regime Detection
    # =========================================================================
    
    def detect_market_regime(
        self,
        prices: np.ndarray,
        volume: np.ndarray,
        lookback: int = 50,
    ) -> Dict[str, Any]:
        """
        Detect the current market regime.
        
        Args:
            prices: Array of closing prices
            volume: Array of volume data
            lookback: Lookback period for regime detection
            
        Returns:
            Dictionary with regime classification and confidence
        """
        if len(prices) < lookback:
            return {"regime": MarketCondition.NEUTRAL, "confidence": 0.0}
        
        recent_prices = prices[-lookback:]
        recent_volume = volume[-lookback:]
        
        # Calculate returns
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Trend metrics
        sma_20 = np.mean(recent_prices[-20:])
        sma_50 = np.mean(recent_prices)
        price_vs_sma20 = (recent_prices[-1] - sma_20) / sma_20
        price_vs_sma50 = (recent_prices[-1] - sma_50) / sma_50
        
        # Volatility metrics
        volatility = np.std(returns)
        volatility_percentile = self._calculate_volatility_percentile(returns)
        
        # Volume metrics
        avg_volume = np.mean(recent_volume)
        recent_avg_volume = np.mean(recent_volume[-10:])
        volume_expansion = recent_avg_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Momentum
        momentum_20 = (recent_prices[-1] - recent_prices[-20]) / recent_prices[-20]
        
        # Classify regime
        regime = MarketCondition.NEUTRAL
        confidence = 0.5
        
        if price_vs_sma20 > 0.05 and price_vs_sma50 > 0.05 and momentum_20 > 0.1:
            if volume_expansion > 1.2:
                regime = MarketCondition.STRONG_BULL
                confidence = 0.85
            else:
                regime = MarketCondition.BULL
                confidence = 0.7
        
        elif price_vs_sma20 < -0.05 and price_vs_sma50 < -0.05 and momentum_20 < -0.1:
            if volume_expansion > 1.2:
                regime = MarketCondition.STRONG_BEAR
                confidence = 0.85
            else:
                regime = MarketCondition.BEAR
                confidence = 0.7
        
        elif volatility_percentile > 80:
            regime = MarketCondition.HIGH_VOLATILITY
            confidence = 0.75
        
        elif volatility_percentile < 20:
            regime = MarketCondition.LOW_VOLATILITY
            confidence = 0.75
        
        elif abs(price_vs_sma20) < 0.02 and abs(momentum_20) < 0.03:
            regime = MarketCondition.CONSOLIDATION
            confidence = 0.65
        
        # Store current regime
        self.current_regime = regime
        self.regime_confidence = confidence
        
        return {
            "regime": regime,
            "confidence": confidence,
            "metrics": {
                "price_vs_sma20": float(price_vs_sma20),
                "price_vs_sma50": float(price_vs_sma50),
                "volatility_percentile": float(volatility_percentile),
                "volume_expansion": float(volume_expansion),
                "momentum_20d": float(momentum_20),
            }
        }

    def _calculate_volatility_percentile(
        self,
        returns: np.ndarray,
        lookback_windows: List[int] = [20, 60, 120],
    ) -> float:
        """Calculate current volatility percentile relative to history."""
        if len(returns) < max(lookback_windows):
            return 50.0
        
        current_vol = np.std(returns[-20:])
        historical_vols = []
        
        for i in range(max(lookback_windows), len(returns)):
            historical_vols.append(np.std(returns[i-20:i]))
        
        if not historical_vols:
            return 50.0
        
        percentile = (np.sum(np.array(historical_vols) <= current_vol) / 
                     len(historical_vols) * 100)
        
        return float(percentile)

    # =========================================================================
    # Signal Aggregation
    # =========================================================================
    
    def aggregate_signals(
        self,
        technical_signals: Dict[str, float],
        fundamental_score: Optional[float] = None,
        sentiment_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate multiple signals into a composite score.
        
        Args:
            technical_signals: Dictionary of technical indicator signals (-1 to 1)
            fundamental_score: Fundamental analysis score (-1 to 1)
            sentiment_score: Market sentiment score (-1 to 1)
            
        Returns:
            Dictionary with composite signal and breakdown
        """
        # Normalize and weight technical signals
        tech_weights = {
            "trend": 0.30,
            "momentum": 0.25,
            "volatility": 0.15,
            "volume": 0.15,
            "pattern": 0.15,
        }
        
        weighted_tech_score = 0.0
        tech_contribution = {}
        
        for signal_type, weight in tech_weights.items():
            if signal_type in technical_signals:
                contribution = technical_signals[signal_type] * weight
                weighted_tech_score += contribution
                tech_contribution[signal_type] = contribution
        
        # Combine all signal types
        composite_score = weighted_tech_score * 0.60
        
        if fundamental_score is not None:
            composite_score += fundamental_score * 0.25
        
        if sentiment_score is not None:
            composite_score += sentiment_score * 0.15
        
        # Determine signal strength
        if composite_score >= 0.6:
            signal = SignalStrength.STRONG_BUY
        elif composite_score >= 0.3:
            signal = SignalStrength.BUY
        elif composite_score >= 0.1:
            signal = SignalStrength.WEAK_BUY
        elif composite_score >= -0.1:
            signal = SignalStrength.NEUTRAL
        elif composite_score >= -0.3:
            signal = SignalStrength.WEAK_SELL
        elif composite_score >= -0.6:
            signal = SignalStrength.SELL
        else:
            signal = SignalStrength.STRONG_SELL
        
        return {
            "composite_score": float(composite_score),
            "signal_strength": signal,
            "technical_contribution": tech_contribution,
            "fundamental_contribution": float(fundamental_score * 0.25) if fundamental_score else 0.0,
            "sentiment_contribution": float(sentiment_score * 0.15) if sentiment_score else 0.0,
            "confidence": abs(composite_score),
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent analysis history."""
        return self._analysis_history[-limit:]

    def run(
        self,
        task: Optional[str] = None,
        symbol: Optional[str] = None,
        price_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Main entry point for running analysis.
        
        Can be called with either:
        - task: A natural language query for the LLM
        - symbol + price_data: Structured data for programmatic analysis
        
        Args:
            task: Natural language analysis request
            symbol: Security symbol for structured analysis
            price_data: Price data dictionary
            **kwargs: Additional arguments
            
        Returns:
            Analysis result dictionary
        """
        if task and not symbol:
            # Natural language query mode
            return {"analysis": super().run(task), "mode": "llm_query"}
        
        if symbol:
            # Structured analysis mode
            return self.analyze_security(
                symbol=symbol,
                price_data=price_data,
                **kwargs,
            )
        
        return {"error": "Please provide either 'task' or 'symbol' parameter"}

