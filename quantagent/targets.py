"""
PriceTargetGenerator - Price Target Generation Module

AI-powered price target generation with multiple methodologies,
confidence intervals, and invalidation tracking.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TargetBasis(Enum):
    """Basis for price target calculation."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    HYBRID = "hybrid"
    FIBONACCI = "fibonacci"
    MEASURED_MOVE = "measured_move"
    VOLUME_PROFILE = "volume_profile"
    REGRESSION = "regression"


class TargetTimeframe(Enum):
    """Target achievement timeframes."""
    INTRADAY = "intraday"
    SHORT_TERM = "1-2_weeks"
    INTERMEDIATE = "4-8_weeks"
    LONG_TERM = "3+_months"


@dataclass
class PriceTarget:
    """Comprehensive price target specification."""
    price: float
    timeframe: TargetTimeframe
    basis: TargetBasis
    confidence: float
    probability: float
    invalidation_price: float
    risk_reward_ratio: float
    supporting_factors: List[str] = field(default_factory=list)
    reasoning: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TargetZone:
    """Price zone target (range rather than point)."""
    lower_bound: float
    upper_bound: float
    midpoint: float
    zone_type: str  # 'accumulation', 'distribution', 'resistance', 'support'
    strength: float
    touches: int = 0


class PriceTargetGenerator:
    """
    Price Target Generation Engine
    
    Generates precise price targets using multiple methodologies:
    - Fibonacci extensions and retracements
    - Measured move projections
    - Volume profile-based targets
    - Regression channel targets
    - Technical pattern targets
    - Fundamental fair value targets
    
    Each target includes confidence levels, invalidation points,
    and risk/reward calculations.
    
    Example:
        >>> generator = PriceTargetGenerator()
        >>> targets = generator.generate_targets(
        ...     current_price=150,
        ...     support_levels=[140, 135, 130],
        ...     resistance_levels=[160, 165, 175],
        ...     atr=2.5,
        ...     trend="bullish"
        ... )
    """
    
    # Fibonacci ratios
    FIB_RETRACEMENTS = [0.236, 0.382, 0.500, 0.618, 0.786]
    FIB_EXTENSIONS = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
    
    def __init__(
        self,
        default_risk_reward_min: float = 1.5,
        confidence_decay_factor: float = 0.85,
    ):
        """
        Initialize the Price Target Generator.
        
        Args:
            default_risk_reward_min: Minimum acceptable risk/reward ratio
            confidence_decay_factor: Decay factor for extended targets
        """
        self.default_risk_reward_min = default_risk_reward_min
        self.confidence_decay_factor = confidence_decay_factor

    def generate_targets(
        self,
        current_price: float,
        support_levels: List[float],
        resistance_levels: List[float],
        atr: float,
        trend: str = "neutral",
        fundamental_fair_value: Optional[float] = None,
        volume_profile: Optional[Dict[str, Any]] = None,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None,
    ) -> List[PriceTarget]:
        """
        Generate comprehensive price targets.
        
        Args:
            current_price: Current market price
            support_levels: Identified support levels
            resistance_levels: Identified resistance levels
            atr: Average True Range
            trend: Current trend direction ('bullish', 'bearish', 'neutral')
            fundamental_fair_value: Fundamental fair value if available
            volume_profile: Volume profile data with POC, VAH, VAL
            swing_high: Recent swing high
            swing_low: Recent swing low
            
        Returns:
            List of PriceTarget objects (typically 3: near, intermediate, extended)
        """
        targets = []
        
        # Generate targets based on trend direction
        if trend == "bullish":
            targets = self._generate_bullish_targets(
                current_price=current_price,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                atr=atr,
                fundamental_fair_value=fundamental_fair_value,
                volume_profile=volume_profile,
                swing_high=swing_high,
                swing_low=swing_low,
            )
        elif trend == "bearish":
            targets = self._generate_bearish_targets(
                current_price=current_price,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                atr=atr,
                fundamental_fair_value=fundamental_fair_value,
                volume_profile=volume_profile,
                swing_high=swing_high,
                swing_low=swing_low,
            )
        else:
            targets = self._generate_neutral_targets(
                current_price=current_price,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                atr=atr,
                volume_profile=volume_profile,
            )
        
        return targets

    def _generate_bullish_targets(
        self,
        current_price: float,
        support_levels: List[float],
        resistance_levels: List[float],
        atr: float,
        fundamental_fair_value: Optional[float] = None,
        volume_profile: Optional[Dict[str, Any]] = None,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None,
    ) -> List[PriceTarget]:
        """Generate bullish price targets."""
        targets = []
        
        # Stop loss level
        if support_levels:
            stop_loss = support_levels[0] - 0.5 * atr
        else:
            stop_loss = current_price - 2 * atr
        
        # Target 1: Near-term (first resistance or 1.5 ATR)
        if resistance_levels:
            target1_price = resistance_levels[0]
            basis = TargetBasis.TECHNICAL
            reasoning = f"First resistance level at ${target1_price:.2f}"
        else:
            target1_price = current_price + 1.5 * atr
            basis = TargetBasis.TECHNICAL
            reasoning = f"1.5x ATR projection from current price"
        
        rr1 = (target1_price - current_price) / (current_price - stop_loss) if current_price != stop_loss else 0
        
        targets.append(PriceTarget(
            price=round(target1_price, 2),
            timeframe=TargetTimeframe.SHORT_TERM,
            basis=basis,
            confidence=0.70,
            probability=0.65,
            invalidation_price=round(stop_loss, 2),
            risk_reward_ratio=round(rr1, 2),
            supporting_factors=["Trend momentum", "First resistance target"],
            reasoning=reasoning,
        ))
        
        # Target 2: Intermediate (second resistance, Fibonacci extension, or volume node)
        target2_candidates = []
        
        if len(resistance_levels) > 1:
            target2_candidates.append((resistance_levels[1], TargetBasis.TECHNICAL, "Second resistance level"))
        
        if swing_low is not None and swing_high is not None:
            fib_ext = self._calculate_fibonacci_extension(swing_low, swing_high, current_price, 1.618)
            if fib_ext > current_price:
                target2_candidates.append((fib_ext, TargetBasis.FIBONACCI, "1.618 Fibonacci extension"))
        
        if volume_profile and "value_area_high" in volume_profile:
            vah = volume_profile["value_area_high"]
            if vah > current_price:
                target2_candidates.append((vah, TargetBasis.VOLUME_PROFILE, "Value Area High"))
        
        if target2_candidates:
            # Choose the most conservative intermediate target
            target2_price, target2_basis, target2_reason = min(target2_candidates, key=lambda x: x[0])
        else:
            target2_price = current_price + 3 * atr
            target2_basis = TargetBasis.TECHNICAL
            target2_reason = "3x ATR projection"
        
        rr2 = (target2_price - current_price) / (current_price - stop_loss) if current_price != stop_loss else 0
        
        targets.append(PriceTarget(
            price=round(target2_price, 2),
            timeframe=TargetTimeframe.INTERMEDIATE,
            basis=target2_basis,
            confidence=0.55,
            probability=0.45,
            invalidation_price=round(stop_loss, 2),
            risk_reward_ratio=round(rr2, 2),
            supporting_factors=["Trend continuation", target2_reason],
            reasoning=target2_reason,
        ))
        
        # Target 3: Extended (fundamental fair value, major resistance, or ambitious technical)
        target3_candidates = []
        
        if fundamental_fair_value and fundamental_fair_value > current_price:
            target3_candidates.append((fundamental_fair_value, TargetBasis.FUNDAMENTAL, "Fundamental fair value estimate"))
        
        if len(resistance_levels) > 2:
            target3_candidates.append((resistance_levels[2], TargetBasis.TECHNICAL, "Major resistance level"))
        
        if swing_low is not None and swing_high is not None:
            fib_ext_261 = self._calculate_fibonacci_extension(swing_low, swing_high, current_price, 2.618)
            if fib_ext_261 > target2_price:
                target3_candidates.append((fib_ext_261, TargetBasis.FIBONACCI, "2.618 Fibonacci extension"))
        
        if target3_candidates:
            target3_price, target3_basis, target3_reason = max(target3_candidates, key=lambda x: x[0])
        else:
            target3_price = current_price + 5 * atr
            target3_basis = TargetBasis.TECHNICAL
            target3_reason = "5x ATR extended projection"
        
        rr3 = (target3_price - current_price) / (current_price - stop_loss) if current_price != stop_loss else 0
        
        targets.append(PriceTarget(
            price=round(target3_price, 2),
            timeframe=TargetTimeframe.LONG_TERM,
            basis=target3_basis,
            confidence=0.40,
            probability=0.30,
            invalidation_price=round(stop_loss, 2),
            risk_reward_ratio=round(rr3, 2),
            supporting_factors=["Extended move potential", target3_reason],
            reasoning=target3_reason,
        ))
        
        return targets

    def _generate_bearish_targets(
        self,
        current_price: float,
        support_levels: List[float],
        resistance_levels: List[float],
        atr: float,
        fundamental_fair_value: Optional[float] = None,
        volume_profile: Optional[Dict[str, Any]] = None,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None,
    ) -> List[PriceTarget]:
        """Generate bearish price targets (downside)."""
        targets = []
        
        # Stop loss level (above resistance)
        if resistance_levels:
            stop_loss = resistance_levels[0] + 0.5 * atr
        else:
            stop_loss = current_price + 2 * atr
        
        # Target 1: Near-term downside
        if support_levels:
            target1_price = support_levels[0]
            basis = TargetBasis.TECHNICAL
            reasoning = f"First support level at ${target1_price:.2f}"
        else:
            target1_price = current_price - 1.5 * atr
            basis = TargetBasis.TECHNICAL
            reasoning = "1.5x ATR downside projection"
        
        rr1 = (current_price - target1_price) / (stop_loss - current_price) if stop_loss != current_price else 0
        
        targets.append(PriceTarget(
            price=round(target1_price, 2),
            timeframe=TargetTimeframe.SHORT_TERM,
            basis=basis,
            confidence=0.70,
            probability=0.65,
            invalidation_price=round(stop_loss, 2),
            risk_reward_ratio=round(rr1, 2),
            supporting_factors=["Downtrend momentum", "First support target"],
            reasoning=reasoning,
        ))
        
        # Target 2: Intermediate downside
        if len(support_levels) > 1:
            target2_price = support_levels[1]
            target2_basis = TargetBasis.TECHNICAL
            target2_reason = "Second support level"
        elif volume_profile and "value_area_low" in volume_profile:
            target2_price = volume_profile["value_area_low"]
            target2_basis = TargetBasis.VOLUME_PROFILE
            target2_reason = "Value Area Low"
        else:
            target2_price = current_price - 3 * atr
            target2_basis = TargetBasis.TECHNICAL
            target2_reason = "3x ATR downside projection"
        
        rr2 = (current_price - target2_price) / (stop_loss - current_price) if stop_loss != current_price else 0
        
        targets.append(PriceTarget(
            price=round(target2_price, 2),
            timeframe=TargetTimeframe.INTERMEDIATE,
            basis=target2_basis,
            confidence=0.55,
            probability=0.45,
            invalidation_price=round(stop_loss, 2),
            risk_reward_ratio=round(rr2, 2),
            supporting_factors=["Trend continuation", target2_reason],
            reasoning=target2_reason,
        ))
        
        # Target 3: Extended downside
        if fundamental_fair_value and fundamental_fair_value < current_price:
            target3_price = fundamental_fair_value
            target3_basis = TargetBasis.FUNDAMENTAL
            target3_reason = "Fundamental fair value (below current)"
        elif len(support_levels) > 2:
            target3_price = support_levels[2]
            target3_basis = TargetBasis.TECHNICAL
            target3_reason = "Major support level"
        else:
            target3_price = current_price - 5 * atr
            target3_basis = TargetBasis.TECHNICAL
            target3_reason = "5x ATR extended downside"
        
        rr3 = (current_price - target3_price) / (stop_loss - current_price) if stop_loss != current_price else 0
        
        targets.append(PriceTarget(
            price=round(target3_price, 2),
            timeframe=TargetTimeframe.LONG_TERM,
            basis=target3_basis,
            confidence=0.40,
            probability=0.30,
            invalidation_price=round(stop_loss, 2),
            risk_reward_ratio=round(rr3, 2),
            supporting_factors=["Extended decline potential", target3_reason],
            reasoning=target3_reason,
        ))
        
        return targets

    def _generate_neutral_targets(
        self,
        current_price: float,
        support_levels: List[float],
        resistance_levels: List[float],
        atr: float,
        volume_profile: Optional[Dict[str, Any]] = None,
    ) -> List[PriceTarget]:
        """Generate targets for neutral/range-bound conditions."""
        targets = []
        
        # Define range bounds
        upper_bound = resistance_levels[0] if resistance_levels else current_price + atr
        lower_bound = support_levels[0] if support_levels else current_price - atr
        
        # Target 1: Upper range (resistance)
        targets.append(PriceTarget(
            price=round(upper_bound, 2),
            timeframe=TargetTimeframe.SHORT_TERM,
            basis=TargetBasis.TECHNICAL,
            confidence=0.60,
            probability=0.50,
            invalidation_price=round(lower_bound - 0.5 * atr, 2),
            risk_reward_ratio=round((upper_bound - current_price) / (current_price - lower_bound), 2) if current_price != lower_bound else 1.0,
            supporting_factors=["Range resistance", "Mean reversion"],
            reasoning="Range high / resistance target",
        ))
        
        # Target 2: Lower range (support)
        targets.append(PriceTarget(
            price=round(lower_bound, 2),
            timeframe=TargetTimeframe.SHORT_TERM,
            basis=TargetBasis.TECHNICAL,
            confidence=0.60,
            probability=0.50,
            invalidation_price=round(upper_bound + 0.5 * atr, 2),
            risk_reward_ratio=round((current_price - lower_bound) / (upper_bound - current_price), 2) if upper_bound != current_price else 1.0,
            supporting_factors=["Range support", "Mean reversion"],
            reasoning="Range low / support target",
        ))
        
        # Target 3: Breakout target (either direction)
        if volume_profile and "poc" in volume_profile:
            poc = volume_profile["poc"]
            targets.append(PriceTarget(
                price=round(poc, 2),
                timeframe=TargetTimeframe.INTERMEDIATE,
                basis=TargetBasis.VOLUME_PROFILE,
                confidence=0.55,
                probability=0.45,
                invalidation_price=round(lower_bound - atr, 2),
                risk_reward_ratio=1.5,
                supporting_factors=["Point of Control", "Volume node"],
                reasoning="Volume POC as potential attractor",
            ))
        else:
            # Potential breakout target
            breakout_target = upper_bound + (upper_bound - lower_bound)  # Measured move
            targets.append(PriceTarget(
                price=round(breakout_target, 2),
                timeframe=TargetTimeframe.INTERMEDIATE,
                basis=TargetBasis.MEASURED_MOVE,
                confidence=0.40,
                probability=0.30,
                invalidation_price=round(lower_bound - 0.5 * atr, 2),
                risk_reward_ratio=2.0,
                supporting_factors=["Potential breakout", "Measured move"],
                reasoning="Breakout measured move target (if range breaks up)",
            ))
        
        return targets

    def _calculate_fibonacci_extension(
        self,
        swing_low: float,
        swing_high: float,
        current_price: float,
        ratio: float,
    ) -> float:
        """Calculate Fibonacci extension level."""
        swing_range = swing_high - swing_low
        extension = swing_high + swing_range * (ratio - 1)
        return extension

    def calculate_fibonacci_levels(
        self,
        high: float,
        low: float,
        trend: str = "bullish",
    ) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement and extension levels.
        
        Args:
            high: Swing high price
            low: Swing low price
            trend: Trend direction for interpretation
            
        Returns:
            Dictionary of Fibonacci levels
        """
        range_size = high - low
        levels = {}
        
        # Retracement levels (in an uptrend, these are potential support)
        for ratio in self.FIB_RETRACEMENTS:
            if trend == "bullish":
                level = high - range_size * ratio
            else:
                level = low + range_size * ratio
            levels[f"retracement_{ratio}"] = round(level, 2)
        
        # Extension levels
        for ratio in self.FIB_EXTENSIONS:
            if trend == "bullish":
                level = high + range_size * (ratio - 1)
            else:
                level = low - range_size * (ratio - 1)
            levels[f"extension_{ratio}"] = round(level, 2)
        
        return levels

    def calculate_measured_move(
        self,
        point_a: float,
        point_b: float,
        point_c: float,
    ) -> Dict[str, float]:
        """
        Calculate measured move (AB=CD) target.
        
        Args:
            point_a: First pivot
            point_b: Second pivot
            point_c: Third pivot (start of projection)
            
        Returns:
            Measured move target and levels
        """
        ab_move = point_b - point_a
        target = point_c + ab_move
        
        return {
            "ab_move": round(ab_move, 2),
            "projected_target": round(target, 2),
            "move_pct": round(abs(ab_move / point_a) * 100, 2) if point_a != 0 else 0,
        }

    def generate_target_zones(
        self,
        targets: List[PriceTarget],
        atr: float,
        zone_width_factor: float = 0.5,
    ) -> List[TargetZone]:
        """
        Convert point targets to zone targets.
        
        Args:
            targets: List of price targets
            atr: Average True Range
            zone_width_factor: Width of zone as multiple of ATR
            
        Returns:
            List of TargetZone objects
        """
        zones = []
        zone_width = atr * zone_width_factor
        
        for target in targets:
            zones.append(TargetZone(
                lower_bound=round(target.price - zone_width / 2, 2),
                upper_bound=round(target.price + zone_width / 2, 2),
                midpoint=target.price,
                zone_type="resistance" if target.price > 0 else "support",  # Simplified
                strength=target.confidence,
            ))
        
        return zones

    def validate_targets(
        self,
        targets: List[PriceTarget],
        current_price: float,
        min_risk_reward: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate and filter targets based on criteria.
        
        Args:
            targets: List of price targets to validate
            current_price: Current market price
            min_risk_reward: Minimum acceptable risk/reward
            
        Returns:
            List of validation results
        """
        min_risk_reward = min_risk_reward or self.default_risk_reward_min
        results = []
        
        for target in targets:
            validation = {
                "target_price": target.price,
                "valid": True,
                "issues": [],
            }
            
            # Check risk/reward
            if target.risk_reward_ratio < min_risk_reward:
                validation["issues"].append(f"R:R below minimum ({target.risk_reward_ratio} < {min_risk_reward})")
            
            # Check target makes sense given direction
            is_bullish = target.price > current_price
            if is_bullish and target.invalidation_price >= current_price:
                validation["issues"].append("Invalidation above current price for bullish target")
            elif not is_bullish and target.invalidation_price <= current_price:
                validation["issues"].append("Invalidation below current price for bearish target")
            
            # Check probability
            if target.probability < 0.20:
                validation["issues"].append("Very low probability target")
            
            validation["valid"] = len(validation["issues"]) == 0
            results.append(validation)
        
        return results

    def calculate_target_probabilities(
        self,
        targets: List[PriceTarget],
        historical_returns: np.ndarray,
        current_price: float,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Calculate probability of reaching targets based on historical data.
        
        Args:
            targets: List of price targets
            historical_returns: Array of historical daily returns
            current_price: Current price
            days: Number of days to simulate
            
        Returns:
            List of probability estimates
        """
        if len(historical_returns) < 30:
            return [{"error": "Insufficient historical data"}]
        
        results = []
        
        # Monte Carlo simulation
        num_simulations = 10000
        mu = np.mean(historical_returns)
        sigma = np.std(historical_returns, ddof=1)
        
        for target in targets:
            target_return = (target.price - current_price) / current_price
            
            # Simulate price paths
            paths_hit_target = 0
            
            for _ in range(num_simulations):
                cumulative_return = 0
                hit_target = False
                hit_invalidation = False
                
                for _ in range(days):
                    daily_return = np.random.normal(mu, sigma)
                    cumulative_return += daily_return
                    
                    simulated_price = current_price * (1 + cumulative_return)
                    
                    # Check if target hit
                    if target.price > current_price:
                        if simulated_price >= target.price:
                            hit_target = True
                            break
                        if simulated_price <= target.invalidation_price:
                            hit_invalidation = True
                            break
                    else:
                        if simulated_price <= target.price:
                            hit_target = True
                            break
                        if simulated_price >= target.invalidation_price:
                            hit_invalidation = True
                            break
                
                if hit_target:
                    paths_hit_target += 1
            
            probability = paths_hit_target / num_simulations
            
            results.append({
                "target_price": target.price,
                "timeframe_days": days,
                "monte_carlo_probability": round(probability, 3),
                "original_probability": target.probability,
                "simulations_run": num_simulations,
            })
        
        return results

