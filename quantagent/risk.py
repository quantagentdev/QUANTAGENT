"""
RiskEngine - Risk Management Module

Comprehensive risk management including VaR calculations, position sizing,
portfolio risk analysis, and drawdown management.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class RiskMetrics:
    """Core risk metrics."""
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    max_drawdown: float
    volatility_annual: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: Optional[float] = None
    alpha: Optional[float] = None


@dataclass
class PositionSize:
    """Position sizing recommendation."""
    shares: float
    position_value: float
    risk_amount: float
    stop_loss_price: float
    position_pct: float
    risk_reward_ratio: float


class RiskEngine:
    """
    Risk Management Engine
    
    Provides comprehensive risk management capabilities including:
    - Multiple VaR methodologies (Historical, Parametric, Monte Carlo)
    - Position sizing algorithms (Fixed Fractional, Kelly, Volatility-adjusted)
    - Portfolio risk analysis
    - Drawdown analysis and recovery
    - Correlation analysis
    - Stress testing
    
    Example:
        >>> engine = RiskEngine(confidence_level=0.95)
        >>> var = engine.calculate_var(returns, method="historical")
        >>> position = engine.calculate_position_size(
        ...     account_value=100000,
        ...     entry_price=150,
        ...     stop_loss=145
        ... )
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.04,
        max_position_size: float = 0.20,
        default_risk_per_trade: float = 0.02,
    ):
        """
        Initialize the Risk Engine.
        
        Args:
            confidence_level: Default confidence level for VaR
            risk_free_rate: Risk-free rate for Sharpe calculations
            max_position_size: Maximum position as fraction of portfolio
            default_risk_per_trade: Default risk per trade as fraction
        """
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.default_risk_per_trade = default_risk_per_trade

    def analyze_risk(
        self,
        returns: np.ndarray,
        prices: Optional[np.ndarray] = None,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis.
        
        Args:
            returns: Array of historical returns
            prices: Array of prices (for drawdown analysis)
            benchmark_returns: Benchmark returns for beta/alpha
            
        Returns:
            Comprehensive risk analysis dictionary
        """
        results = {}
        
        # VaR calculations
        results["var"] = {
            "historical_95": self.calculate_var(returns, 0.95, "historical"),
            "historical_99": self.calculate_var(returns, 0.99, "historical"),
            "parametric_95": self.calculate_var(returns, 0.95, "parametric"),
            "cornish_fisher_95": self.calculate_var(returns, 0.95, "cornish_fisher"),
        }
        
        # CVaR
        results["cvar"] = {
            "cvar_95": self.calculate_cvar(returns, 0.95),
            "cvar_99": self.calculate_cvar(returns, 0.99),
        }
        
        # Volatility metrics
        results["volatility"] = self._analyze_volatility(returns)
        
        # Performance metrics
        results["performance"] = {
            "sharpe_ratio": self.calculate_sharpe(returns),
            "sortino_ratio": self.calculate_sortino(returns),
            "information_ratio": self.calculate_information_ratio(
                returns, benchmark_returns
            ) if benchmark_returns is not None else None,
        }
        
        # Drawdown analysis
        if prices is not None:
            results["drawdown"] = self.analyze_drawdown(prices)
            results["performance"]["calmar_ratio"] = self.calculate_calmar(
                returns, prices
            )
        
        # Beta and Alpha
        if benchmark_returns is not None:
            beta_alpha = self.calculate_beta_alpha(returns, benchmark_returns)
            results["market_risk"] = beta_alpha
        
        # Risk classification
        results["risk_level"] = self._classify_risk(results)
        
        return results

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: Optional[float] = None,
        method: str = "historical",
        holding_period: int = 1,
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (default: instance setting)
            method: Calculation method ('historical', 'parametric', 'cornish_fisher', 'monte_carlo')
            holding_period: Holding period in days
            
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
            sigma = np.std(returns, ddof=1)
            var = -(mu + sigma * norm.ppf(1 - confidence_level))
        
        elif method == "cornish_fisher":
            # Cornish-Fisher expansion for non-normal distributions
            from scipy.stats import norm
            mu = np.mean(returns)
            sigma = np.std(returns, ddof=1)
            skew = self._skewness(returns)
            kurt = self._kurtosis(returns)
            
            z = norm.ppf(1 - confidence_level)
            z_cf = (z + (z**2 - 1) * skew / 6 + 
                    (z**3 - 3*z) * kurt / 24 - 
                    (2*z**3 - 5*z) * skew**2 / 36)
            
            var = -(mu + sigma * z_cf)
        
        elif method == "monte_carlo":
            var = self._monte_carlo_var(returns, confidence_level)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # Scale for holding period
        var = var * np.sqrt(holding_period)
        
        return float(max(0, var))

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
        return float(max(0, cvar))

    def _monte_carlo_var(
        self,
        returns: np.ndarray,
        confidence_level: float,
        num_simulations: int = 10000,
    ) -> float:
        """Monte Carlo VaR simulation."""
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        
        simulated = np.random.normal(mu, sigma, num_simulations)
        var = -np.percentile(simulated, (1 - confidence_level) * 100)
        
        return float(var)

    def calculate_sharpe(
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
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
        return float(sharpe * np.sqrt(annualization_factor))

    def calculate_sortino(
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
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else np.nan
        
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return np.nan
        
        sortino = np.mean(excess_returns) / downside_std
        return float(sortino * np.sqrt(annualization_factor))

    def calculate_calmar(
        self,
        returns: np.ndarray,
        prices: np.ndarray,
        annualization_factor: int = 252,
    ) -> float:
        """
        Calculate Calmar Ratio (return / max drawdown).
        
        Args:
            returns: Array of returns
            prices: Array of prices
            annualization_factor: Trading days per year
            
        Returns:
            Calmar Ratio
        """
        annual_return = np.mean(returns) * annualization_factor
        max_dd, _, _ = self.calculate_max_drawdown(prices)
        
        if max_dd == 0:
            return np.nan
        
        return float(-annual_return / max_dd)

    def calculate_information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        annualization_factor: int = 252,
    ) -> float:
        """
        Calculate Information Ratio.
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            annualization_factor: Trading days per year
            
        Returns:
            Information Ratio
        """
        if len(returns) != len(benchmark_returns):
            min_len = min(len(returns), len(benchmark_returns))
            returns = returns[-min_len:]
            benchmark_returns = benchmark_returns[-min_len:]
        
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns, ddof=1)
        
        if tracking_error == 0:
            return np.nan
        
        ir = np.mean(active_returns) / tracking_error
        return float(ir * np.sqrt(annualization_factor))

    def calculate_beta_alpha(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate Beta and Alpha using linear regression.
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary with beta, alpha, and r-squared
        """
        if len(returns) != len(benchmark_returns):
            min_len = min(len(returns), len(benchmark_returns))
            returns = returns[-min_len:]
            benchmark_returns = benchmark_returns[-min_len:]
        
        # Linear regression: y = alpha + beta * x
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)
        
        if benchmark_variance == 0:
            return {"beta": np.nan, "alpha": np.nan, "r_squared": np.nan}
        
        beta = covariance / benchmark_variance
        alpha = np.mean(returns) - beta * np.mean(benchmark_returns)
        
        # R-squared
        predicted = alpha + beta * benchmark_returns
        ss_res = np.sum((returns - predicted) ** 2)
        ss_tot = np.sum((returns - np.mean(returns)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "beta": float(beta),
            "alpha": float(alpha * 252),  # Annualized alpha
            "r_squared": float(r_squared),
        }

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
        peak_idx = int(np.argmax(prices[:trough_idx+1])) if trough_idx > 0 else 0
        
        return (max_dd, peak_idx, trough_idx)

    def analyze_drawdown(
        self,
        prices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Comprehensive drawdown analysis.
        
        Args:
            prices: Array of prices
            
        Returns:
            Drawdown analysis dictionary
        """
        if len(prices) < 2:
            return {"error": "Insufficient data"}
        
        cumulative_max = np.maximum.accumulate(prices)
        drawdowns = (prices - cumulative_max) / cumulative_max
        
        max_dd, peak_idx, trough_idx = self.calculate_max_drawdown(prices)
        
        # Current drawdown
        current_dd = float(drawdowns[-1])
        
        # Drawdown statistics
        dd_periods = []
        in_drawdown = False
        dd_start = 0
        
        for i, dd in enumerate(drawdowns):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                dd_start = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                dd_periods.append({
                    "start": dd_start,
                    "end": i,
                    "duration": i - dd_start,
                    "depth": float(np.min(drawdowns[dd_start:i])),
                })
        
        # If still in drawdown
        if in_drawdown:
            dd_periods.append({
                "start": dd_start,
                "end": len(drawdowns) - 1,
                "duration": len(drawdowns) - 1 - dd_start,
                "depth": float(np.min(drawdowns[dd_start:])),
            })
        
        # Recovery analysis
        if current_dd < 0:
            recovery_needed = (prices[-1] / cumulative_max[-1] - 1) * -100
        else:
            recovery_needed = 0
        
        return {
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd * 100,
            "current_drawdown": current_dd,
            "current_drawdown_pct": current_dd * 100,
            "peak_to_trough_days": trough_idx - peak_idx if trough_idx > peak_idx else 0,
            "recovery_needed_pct": recovery_needed,
            "num_drawdown_periods": len(dd_periods),
            "avg_drawdown_duration": np.mean([d["duration"] for d in dd_periods]) if dd_periods else 0,
            "worst_drawdowns": sorted(dd_periods, key=lambda x: x["depth"])[:5],
        }

    def _analyze_volatility(
        self,
        returns: np.ndarray,
        annualization_factor: int = 252,
    ) -> Dict[str, Any]:
        """Analyze volatility metrics."""
        daily_vol = np.std(returns, ddof=1)
        annual_vol = daily_vol * np.sqrt(annualization_factor)
        
        # Rolling volatility
        window = min(20, len(returns) // 2)
        rolling_vol = []
        for i in range(window, len(returns)):
            rolling_vol.append(np.std(returns[i-window:i], ddof=1))
        
        # Volatility regime
        if annual_vol > 0.40:
            regime = "very_high"
        elif annual_vol > 0.25:
            regime = "high"
        elif annual_vol > 0.15:
            regime = "moderate"
        elif annual_vol > 0.08:
            regime = "low"
        else:
            regime = "very_low"
        
        return {
            "daily": float(daily_vol),
            "annual": float(annual_vol),
            "annual_pct": float(annual_vol * 100),
            "current_vs_avg": float(rolling_vol[-1] / np.mean(rolling_vol)) if rolling_vol else 1.0,
            "regime": regime,
            "percentile": float(np.percentile(rolling_vol, 50)) if rolling_vol else daily_vol,
        }

    def calculate_position_size(
        self,
        account_value: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade: Optional[float] = None,
        target_price: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size using fixed fractional risk.
        
        Args:
            account_value: Total account value
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            risk_per_trade: Maximum risk per trade as fraction
            target_price: Optional target price for R:R calculation
            volatility: Optional volatility for adjustment
            
        Returns:
            Position sizing recommendation dictionary
        """
        risk_per_trade = risk_per_trade or self.default_risk_per_trade
        
        # Maximum risk amount
        risk_amount = account_value * risk_per_trade
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return {"error": "Stop loss cannot equal entry price"}
        
        # Calculate position size
        shares = risk_amount / risk_per_share
        position_value = shares * entry_price
        position_pct = position_value / account_value
        
        # Apply maximum position constraint
        if position_pct > self.max_position_size:
            position_value = account_value * self.max_position_size
            shares = position_value / entry_price
            position_pct = self.max_position_size
            actual_risk_pct = (shares * risk_per_share) / account_value
        else:
            actual_risk_pct = risk_per_trade
        
        # Volatility adjustment
        if volatility is not None:
            volatility_factor = 0.20 / volatility  # Normalize to 20% vol
            volatility_factor = max(0.5, min(2.0, volatility_factor))
            shares *= volatility_factor
            position_value = shares * entry_price
            position_pct = position_value / account_value
        
        # Risk/reward ratio
        if target_price is not None:
            potential_gain = abs(target_price - entry_price)
            risk_reward = potential_gain / risk_per_share
        else:
            risk_reward = None
        
        return {
            "shares": round(shares, 2),
            "position_value": round(position_value, 2),
            "position_pct": round(position_pct * 100, 2),
            "risk_amount": round(shares * risk_per_share, 2),
            "risk_pct": round(actual_risk_pct * 100, 2),
            "risk_per_share": round(risk_per_share, 2),
            "stop_loss": stop_loss_price,
            "risk_reward_ratio": round(risk_reward, 2) if risk_reward else None,
            "entry_price": entry_price,
            "target_price": target_price,
        }

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade return (as decimal)
            avg_loss: Average losing trade return (positive number, as decimal)
            kelly_fraction: Fraction of full Kelly to use (default: 0.25)
            
        Returns:
            Kelly criterion result dictionary
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return {"error": "Invalid inputs for Kelly calculation"}
        
        # Kelly formula: f* = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - p
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        full_kelly = (b * p - q) / b
        
        # Apply fractional Kelly for safety
        fractional_kelly = full_kelly * kelly_fraction
        
        # Bounds
        optimal_fraction = max(0, min(self.max_position_size, fractional_kelly))
        
        return {
            "full_kelly": round(full_kelly * 100, 2),
            "fractional_kelly": round(fractional_kelly * 100, 2),
            "recommended_position_pct": round(optimal_fraction * 100, 2),
            "kelly_fraction_used": kelly_fraction,
            "win_rate": win_rate,
            "payoff_ratio": round(b, 2),
            "edge": round((win_rate * avg_win - q * avg_loss) * 100, 2),
        }

    def volatility_adjusted_position(
        self,
        account_value: float,
        entry_price: float,
        volatility: float,
        target_risk_pct: float = 0.02,
        vol_lookback_factor: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Calculate position size based on volatility targeting.
        
        Args:
            account_value: Total account value
            entry_price: Entry price
            volatility: Daily volatility of the asset
            target_risk_pct: Target daily risk as percentage
            vol_lookback_factor: Multiplier for stop loss (x * vol)
            
        Returns:
            Volatility-adjusted position sizing dictionary
        """
        # Target dollar risk
        target_risk = account_value * target_risk_pct
        
        # Calculate position size to achieve target risk
        # Assuming daily move = entry_price * volatility
        daily_dollar_move = entry_price * volatility
        
        shares = target_risk / daily_dollar_move
        position_value = shares * entry_price
        
        # Implied stop loss
        stop_loss_distance = entry_price * volatility * vol_lookback_factor
        stop_loss = entry_price - stop_loss_distance
        
        # Position percentage
        position_pct = position_value / account_value
        
        # Cap at max position
        if position_pct > self.max_position_size:
            position_pct = self.max_position_size
            position_value = account_value * position_pct
            shares = position_value / entry_price
        
        return {
            "shares": round(shares, 2),
            "position_value": round(position_value, 2),
            "position_pct": round(position_pct * 100, 2),
            "implied_stop_loss": round(stop_loss, 2),
            "expected_daily_move": round(daily_dollar_move * shares, 2),
            "target_daily_risk": round(target_risk, 2),
            "volatility_used": round(volatility * 100, 2),
        }

    def stress_test(
        self,
        returns: np.ndarray,
        scenarios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Perform stress testing under various scenarios.
        
        Args:
            returns: Historical returns
            scenarios: Dictionary of scenario names to market moves
            
        Returns:
            Stress test results dictionary
        """
        if scenarios is None:
            scenarios = {
                "2008_financial_crisis": -0.35,
                "2020_covid_crash": -0.30,
                "black_monday_1987": -0.22,
                "flash_crash": -0.10,
                "mild_correction": -0.10,
                "moderate_correction": -0.15,
                "bear_market": -0.25,
            }
        
        # Calculate beta to estimate scenario impact
        # Assuming beta = 1 for simplicity; could be parameterized
        beta = 1.0
        
        results = {}
        for scenario_name, market_move in scenarios.items():
            # Expected portfolio move = beta * market move
            portfolio_impact = beta * market_move
            
            # Probability of similar move based on historical data
            if len(returns) > 0:
                extreme_returns = returns[returns <= market_move]
                probability = len(extreme_returns) / len(returns)
            else:
                probability = None
            
            results[scenario_name] = {
                "market_move_pct": market_move * 100,
                "expected_portfolio_impact_pct": round(portfolio_impact * 100, 2),
                "historical_probability": round(probability * 100, 2) if probability else None,
            }
        
        # Tail risk analysis
        if len(returns) > 0:
            worst_1pct = np.percentile(returns, 1)
            worst_5pct = np.percentile(returns, 5)
            
            results["tail_risk"] = {
                "worst_1_percent_day": round(worst_1pct * 100, 2),
                "worst_5_percent_day": round(worst_5pct * 100, 2),
                "worst_day_in_history": round(np.min(returns) * 100, 2),
                "best_day_in_history": round(np.max(returns) * 100, 2),
            }
        
        return results

    def _classify_risk(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Classify overall risk level."""
        scores = []
        
        # VaR-based risk
        var_95 = results.get("var", {}).get("historical_95", 0)
        if var_95 < 0.01:
            scores.append(1)
        elif var_95 < 0.02:
            scores.append(2)
        elif var_95 < 0.03:
            scores.append(3)
        elif var_95 < 0.05:
            scores.append(4)
        else:
            scores.append(5)
        
        # Volatility-based risk
        vol_annual = results.get("volatility", {}).get("annual", 0)
        if vol_annual < 0.10:
            scores.append(1)
        elif vol_annual < 0.20:
            scores.append(2)
        elif vol_annual < 0.30:
            scores.append(3)
        elif vol_annual < 0.50:
            scores.append(4)
        else:
            scores.append(5)
        
        # Drawdown-based risk
        max_dd = abs(results.get("drawdown", {}).get("max_drawdown", 0))
        if max_dd < 0.10:
            scores.append(1)
        elif max_dd < 0.20:
            scores.append(2)
        elif max_dd < 0.30:
            scores.append(3)
        elif max_dd < 0.50:
            scores.append(4)
        else:
            scores.append(5)
        
        avg_score = np.mean(scores) if scores else 3
        
        if avg_score <= 1.5:
            level = RiskLevel.VERY_LOW
        elif avg_score <= 2.5:
            level = RiskLevel.LOW
        elif avg_score <= 3.5:
            level = RiskLevel.MODERATE
        elif avg_score <= 4.5:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.VERY_HIGH
        
        return {
            "level": level.value,
            "score": float(avg_score),
            "components": {
                "var_risk": scores[0] if len(scores) > 0 else None,
                "volatility_risk": scores[1] if len(scores) > 1 else None,
                "drawdown_risk": scores[2] if len(scores) > 2 else None,
            },
        }

    def _skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness."""
        n = len(returns)
        if n < 3:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std == 0:
            return 0.0
        return float(n / ((n-1) * (n-2)) * np.sum(((returns - mean) / std) ** 3))

    def _kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        n = len(returns)
        if n < 4:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std == 0:
            return 0.0
        kurt = np.mean(((returns - mean) / std) ** 4) - 3
        return float(kurt)

