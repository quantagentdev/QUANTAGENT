"""
FundamentalEngine - Fundamental Analysis Module

Comprehensive fundamental analysis including valuation models, financial
statement analysis, and intrinsic value calculations.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ValuationMethod(Enum):
    """Valuation methodology types."""
    DCF = "discounted_cash_flow"
    DDM = "dividend_discount_model"
    PE_RELATIVE = "pe_relative"
    PB_RELATIVE = "pb_relative"
    EV_EBITDA = "ev_ebitda"
    REGRESSION = "regression_based"
    SUM_OF_PARTS = "sum_of_parts"


class FinancialHealth(Enum):
    """Financial health classifications."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    DISTRESSED = "distressed"


@dataclass
class ValuationResult:
    """Valuation analysis result."""
    method: ValuationMethod
    fair_value: float
    current_price: float
    upside_potential: float
    confidence: float
    assumptions: Dict[str, Any]
    sensitivity: Dict[str, float]


@dataclass
class FinancialMetrics:
    """Core financial metrics."""
    # Profitability
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    roic: Optional[float] = None  # Return on Invested Capital
    
    # Liquidity
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    
    # Leverage
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    interest_coverage: Optional[float] = None
    
    # Efficiency
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_turnover: Optional[float] = None
    
    # Valuation
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    peg_ratio: Optional[float] = None
    
    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    fcf_growth: Optional[float] = None


class FundamentalEngine:
    """
    Fundamental Analysis Engine
    
    Provides comprehensive fundamental analysis capabilities including:
    - Multiple valuation methodologies (DCF, DDM, relative)
    - Financial statement analysis
    - Ratio analysis and scoring
    - Peer comparison
    - Quality scoring
    - Fair value estimation with confidence intervals
    
    Example:
        >>> engine = FundamentalEngine()
        >>> valuation = engine.dcf_valuation(
        ...     fcf=[100, 110, 120],
        ...     growth_rate=0.05,
        ...     terminal_growth=0.02,
        ...     wacc=0.08
        ... )
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.04,
        equity_risk_premium: float = 0.05,
        default_terminal_growth: float = 0.025,
    ):
        """
        Initialize the Fundamental Analysis Engine.
        
        Args:
            risk_free_rate: Current risk-free rate
            equity_risk_premium: Equity risk premium for CAPM
            default_terminal_growth: Default terminal growth rate
        """
        self.risk_free_rate = risk_free_rate
        self.equity_risk_premium = equity_risk_premium
        self.default_terminal_growth = default_terminal_growth

    def analyze(
        self,
        financials: Dict[str, Any],
        current_price: float,
        shares_outstanding: float,
        sector: Optional[str] = None,
        peer_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis.
        
        Args:
            financials: Financial statement data
            current_price: Current market price
            shares_outstanding: Number of shares outstanding
            sector: Company sector for context
            peer_metrics: Peer comparison metrics
            
        Returns:
            Comprehensive fundamental analysis dictionary
        """
        results = {}
        
        # Calculate financial ratios
        results["ratios"] = self.calculate_ratios(financials)
        
        # Financial health assessment
        results["health"] = self.assess_financial_health(results["ratios"])
        
        # Quality score
        results["quality_score"] = self.calculate_quality_score(
            results["ratios"], 
            financials
        )
        
        # Valuations
        market_cap = current_price * shares_outstanding
        
        if "free_cash_flow" in financials:
            fcf = financials["free_cash_flow"]
            if isinstance(fcf, list):
                results["dcf_valuation"] = self.dcf_valuation(
                    fcf=fcf,
                    shares_outstanding=shares_outstanding,
                    current_price=current_price,
                )
        
        # Relative valuation
        if peer_metrics:
            results["relative_valuation"] = self.relative_valuation(
                financials=financials,
                current_price=current_price,
                shares_outstanding=shares_outstanding,
                peer_metrics=peer_metrics,
            )
        
        # Earnings power value
        if "operating_income" in financials:
            results["earnings_power"] = self.earnings_power_value(
                operating_income=financials["operating_income"],
                tax_rate=financials.get("tax_rate", 0.25),
                wacc=self._calculate_wacc(financials),
                shares_outstanding=shares_outstanding,
            )
        
        # Generate fair value estimate
        results["fair_value_estimate"] = self._synthesize_fair_value(results, current_price)
        
        return results

    def calculate_ratios(
        self,
        financials: Dict[str, Any],
    ) -> FinancialMetrics:
        """
        Calculate comprehensive financial ratios.
        
        Args:
            financials: Financial statement data
            
        Returns:
            FinancialMetrics dataclass with calculated ratios
        """
        metrics = FinancialMetrics()
        
        # Extract values with defaults
        revenue = financials.get("revenue", 0)
        gross_profit = financials.get("gross_profit", 0)
        operating_income = financials.get("operating_income", 0)
        net_income = financials.get("net_income", 0)
        total_assets = financials.get("total_assets", 1)
        total_equity = financials.get("total_equity", 1)
        total_debt = financials.get("total_debt", 0)
        current_assets = financials.get("current_assets", 0)
        current_liabilities = financials.get("current_liabilities", 1)
        inventory = financials.get("inventory", 0)
        cash = financials.get("cash", 0)
        interest_expense = financials.get("interest_expense", 0)
        
        # Profitability ratios
        if revenue > 0:
            metrics.gross_margin = gross_profit / revenue
            metrics.operating_margin = operating_income / revenue
            metrics.net_margin = net_income / revenue
        
        if total_equity > 0:
            metrics.roe = net_income / total_equity
        
        if total_assets > 0:
            metrics.roa = net_income / total_assets
        
        # ROIC
        invested_capital = total_equity + total_debt - cash
        if invested_capital > 0:
            nopat = operating_income * (1 - financials.get("tax_rate", 0.25))
            metrics.roic = nopat / invested_capital
        
        # Liquidity ratios
        if current_liabilities > 0:
            metrics.current_ratio = current_assets / current_liabilities
            metrics.quick_ratio = (current_assets - inventory) / current_liabilities
            metrics.cash_ratio = cash / current_liabilities
        
        # Leverage ratios
        if total_equity > 0:
            metrics.debt_to_equity = total_debt / total_equity
        
        if total_assets > 0:
            metrics.debt_to_assets = total_debt / total_assets
        
        if interest_expense > 0:
            metrics.interest_coverage = operating_income / interest_expense
        
        # Efficiency ratios
        if total_assets > 0:
            metrics.asset_turnover = revenue / total_assets
        
        cogs = financials.get("cost_of_goods_sold", revenue - gross_profit)
        if inventory > 0 and cogs > 0:
            metrics.asset_turnover = cogs / inventory
        
        # Valuation ratios
        pe = financials.get("pe_ratio")
        if pe:
            metrics.pe_ratio = pe
        
        pb = financials.get("pb_ratio")
        if pb:
            metrics.pb_ratio = pb
        
        ev_ebitda = financials.get("ev_ebitda")
        if ev_ebitda:
            metrics.ev_ebitda = ev_ebitda
        
        # Growth rates
        metrics.revenue_growth = financials.get("revenue_growth")
        metrics.earnings_growth = financials.get("earnings_growth")
        metrics.fcf_growth = financials.get("fcf_growth")
        
        return metrics

    def assess_financial_health(
        self,
        metrics: FinancialMetrics,
    ) -> Dict[str, Any]:
        """
        Assess overall financial health.
        
        Args:
            metrics: Calculated financial metrics
            
        Returns:
            Financial health assessment dictionary
        """
        scores = []
        concerns = []
        strengths = []
        
        # Profitability assessment
        if metrics.net_margin is not None:
            if metrics.net_margin > 0.15:
                scores.append(5)
                strengths.append("Strong net margin")
            elif metrics.net_margin > 0.05:
                scores.append(3)
            elif metrics.net_margin > 0:
                scores.append(2)
            else:
                scores.append(0)
                concerns.append("Negative net margin")
        
        if metrics.roe is not None:
            if metrics.roe > 0.15:
                scores.append(5)
                strengths.append("High return on equity")
            elif metrics.roe > 0.10:
                scores.append(4)
            elif metrics.roe > 0.05:
                scores.append(3)
            elif metrics.roe > 0:
                scores.append(2)
            else:
                scores.append(0)
                concerns.append("Negative ROE")
        
        # Liquidity assessment
        if metrics.current_ratio is not None:
            if metrics.current_ratio > 2.0:
                scores.append(5)
            elif metrics.current_ratio > 1.5:
                scores.append(4)
            elif metrics.current_ratio > 1.0:
                scores.append(3)
            else:
                scores.append(1)
                concerns.append("Low current ratio")
        
        # Leverage assessment
        if metrics.debt_to_equity is not None:
            if metrics.debt_to_equity < 0.3:
                scores.append(5)
                strengths.append("Low debt levels")
            elif metrics.debt_to_equity < 0.6:
                scores.append(4)
            elif metrics.debt_to_equity < 1.0:
                scores.append(3)
            elif metrics.debt_to_equity < 2.0:
                scores.append(2)
            else:
                scores.append(1)
                concerns.append("High debt-to-equity")
        
        if metrics.interest_coverage is not None:
            if metrics.interest_coverage > 10:
                scores.append(5)
            elif metrics.interest_coverage > 5:
                scores.append(4)
            elif metrics.interest_coverage > 2:
                scores.append(3)
            elif metrics.interest_coverage > 1:
                scores.append(2)
            else:
                scores.append(0)
                concerns.append("Low interest coverage")
        
        # Calculate overall score
        avg_score = np.mean(scores) if scores else 2.5
        
        if avg_score >= 4.0:
            health = FinancialHealth.EXCELLENT
        elif avg_score >= 3.0:
            health = FinancialHealth.GOOD
        elif avg_score >= 2.0:
            health = FinancialHealth.FAIR
        elif avg_score >= 1.0:
            health = FinancialHealth.POOR
        else:
            health = FinancialHealth.DISTRESSED
        
        return {
            "health_rating": health.value,
            "score": float(avg_score),
            "strengths": strengths,
            "concerns": concerns,
            "component_scores": scores,
        }

    def calculate_quality_score(
        self,
        metrics: FinancialMetrics,
        financials: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate quality score based on multiple factors.
        
        Args:
            metrics: Financial metrics
            financials: Raw financial data
            
        Returns:
            Quality score dictionary
        """
        score_components = {}
        
        # Profitability quality (0-25 points)
        prof_score = 0
        if metrics.roic is not None and metrics.roic > 0.10:
            prof_score += 10
        if metrics.roe is not None and metrics.roe > 0.12:
            prof_score += 8
        if metrics.operating_margin is not None and metrics.operating_margin > 0.10:
            prof_score += 7
        score_components["profitability"] = min(25, prof_score)
        
        # Balance sheet quality (0-25 points)
        bs_score = 0
        if metrics.debt_to_equity is not None and metrics.debt_to_equity < 0.5:
            bs_score += 10
        if metrics.current_ratio is not None and metrics.current_ratio > 1.5:
            bs_score += 8
        if metrics.interest_coverage is not None and metrics.interest_coverage > 5:
            bs_score += 7
        score_components["balance_sheet"] = min(25, bs_score)
        
        # Growth quality (0-25 points)
        growth_score = 0
        if metrics.revenue_growth is not None and metrics.revenue_growth > 0.05:
            growth_score += 8
        if metrics.earnings_growth is not None and metrics.earnings_growth > 0.08:
            growth_score += 8
        if metrics.fcf_growth is not None and metrics.fcf_growth > 0.05:
            growth_score += 9
        score_components["growth"] = min(25, growth_score)
        
        # Cash flow quality (0-25 points)
        cf_score = 0
        fcf = financials.get("free_cash_flow")
        net_income = financials.get("net_income")
        if fcf is not None and net_income is not None and net_income > 0:
            if isinstance(fcf, list):
                fcf = fcf[-1]
            if fcf > 0 and fcf / net_income > 0.8:
                cf_score += 15
        
        operating_cf = financials.get("operating_cash_flow")
        if operating_cf is not None and operating_cf > 0:
            cf_score += 10
        score_components["cash_flow"] = min(25, cf_score)
        
        total_score = sum(score_components.values())
        
        # Quality grade
        if total_score >= 80:
            grade = "A"
        elif total_score >= 60:
            grade = "B"
        elif total_score >= 40:
            grade = "C"
        elif total_score >= 20:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "total_score": total_score,
            "grade": grade,
            "components": score_components,
            "max_score": 100,
        }

    def dcf_valuation(
        self,
        fcf: List[float],
        shares_outstanding: float,
        current_price: float,
        growth_rate: Optional[float] = None,
        terminal_growth: Optional[float] = None,
        wacc: Optional[float] = None,
        projection_years: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform DCF valuation.
        
        Args:
            fcf: Historical free cash flow (most recent last)
            shares_outstanding: Number of shares outstanding
            current_price: Current market price
            growth_rate: Expected FCF growth rate
            terminal_growth: Terminal growth rate
            wacc: Weighted average cost of capital
            projection_years: Number of years to project
            
        Returns:
            DCF valuation result dictionary
        """
        # Default assumptions
        if growth_rate is None:
            if len(fcf) >= 2:
                # Calculate historical growth
                growth_rates = []
                for i in range(1, len(fcf)):
                    if fcf[i-1] > 0:
                        growth_rates.append((fcf[i] - fcf[i-1]) / fcf[i-1])
                growth_rate = np.mean(growth_rates) if growth_rates else 0.05
            else:
                growth_rate = 0.05
        
        terminal_growth = terminal_growth or self.default_terminal_growth
        wacc = wacc or 0.10
        
        # Ensure reasonable bounds
        growth_rate = max(-0.10, min(0.30, growth_rate))
        terminal_growth = max(0.01, min(0.04, terminal_growth))
        
        # Project future FCF
        current_fcf = fcf[-1] if isinstance(fcf, list) else fcf
        projected_fcf = []
        
        for year in range(1, projection_years + 1):
            projected_fcf.append(current_fcf * (1 + growth_rate) ** year)
        
        # Calculate present value of projected FCF
        pv_fcf = []
        for year, cf in enumerate(projected_fcf, 1):
            pv_fcf.append(cf / (1 + wacc) ** year)
        
        # Terminal value
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        pv_terminal = terminal_value / (1 + wacc) ** projection_years
        
        # Enterprise value
        enterprise_value = sum(pv_fcf) + pv_terminal
        
        # Equity value per share
        fair_value_per_share = enterprise_value / shares_outstanding
        
        # Upside/downside
        upside = (fair_value_per_share - current_price) / current_price * 100
        
        # Sensitivity analysis
        sensitivity = {}
        for wacc_adj in [-0.01, 0, 0.01]:
            for growth_adj in [-0.01, 0, 0.01]:
                adj_wacc = wacc + wacc_adj
                adj_terminal = terminal_growth + growth_adj
                
                if adj_wacc > adj_terminal:
                    adj_terminal_value = (projected_fcf[-1] * (1 + adj_terminal)) / (adj_wacc - adj_terminal)
                    adj_pv_terminal = adj_terminal_value / (1 + adj_wacc) ** projection_years
                    adj_pv_fcf = [cf / (1 + adj_wacc) ** (y+1) for y, cf in enumerate(projected_fcf)]
                    adj_ev = sum(adj_pv_fcf) + adj_pv_terminal
                    adj_fair_value = adj_ev / shares_outstanding
                    sensitivity[f"wacc_{adj_wacc:.1%}_growth_{adj_terminal:.1%}"] = round(adj_fair_value, 2)
        
        return {
            "method": ValuationMethod.DCF.value,
            "fair_value": round(fair_value_per_share, 2),
            "current_price": current_price,
            "upside_percent": round(upside, 2),
            "assumptions": {
                "wacc": wacc,
                "growth_rate": growth_rate,
                "terminal_growth": terminal_growth,
                "projection_years": projection_years,
            },
            "projected_fcf": [round(f, 2) for f in projected_fcf],
            "terminal_value": round(terminal_value, 2),
            "enterprise_value": round(enterprise_value, 2),
            "sensitivity": sensitivity,
            "confidence": 0.7 if abs(upside) < 50 else 0.5,
        }

    def relative_valuation(
        self,
        financials: Dict[str, Any],
        current_price: float,
        shares_outstanding: float,
        peer_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Perform relative valuation using peer comparisons.
        
        Args:
            financials: Company financial data
            current_price: Current market price
            shares_outstanding: Number of shares outstanding
            peer_metrics: Dictionary of peer average metrics
            
        Returns:
            Relative valuation result dictionary
        """
        valuations = {}
        market_cap = current_price * shares_outstanding
        
        # P/E based valuation
        if "eps" in financials and "pe_ratio" in peer_metrics:
            eps = financials["eps"]
            peer_pe = peer_metrics["pe_ratio"]
            if eps > 0:
                implied_price = eps * peer_pe
                valuations["pe_based"] = {
                    "fair_value": round(implied_price, 2),
                    "peer_pe": peer_pe,
                    "company_pe": current_price / eps,
                    "premium_discount": round((current_price - implied_price) / implied_price * 100, 2),
                }
        
        # P/B based valuation
        if "book_value_per_share" in financials and "pb_ratio" in peer_metrics:
            bvps = financials["book_value_per_share"]
            peer_pb = peer_metrics["pb_ratio"]
            if bvps > 0:
                implied_price = bvps * peer_pb
                valuations["pb_based"] = {
                    "fair_value": round(implied_price, 2),
                    "peer_pb": peer_pb,
                    "company_pb": current_price / bvps,
                    "premium_discount": round((current_price - implied_price) / implied_price * 100, 2),
                }
        
        # EV/EBITDA based valuation
        if "ebitda" in financials and "ev_ebitda" in peer_metrics:
            ebitda = financials["ebitda"]
            peer_ev_ebitda = peer_metrics["ev_ebitda"]
            net_debt = financials.get("total_debt", 0) - financials.get("cash", 0)
            
            if ebitda > 0:
                implied_ev = ebitda * peer_ev_ebitda
                implied_equity = implied_ev - net_debt
                implied_price = implied_equity / shares_outstanding
                
                valuations["ev_ebitda_based"] = {
                    "fair_value": round(implied_price, 2),
                    "peer_ev_ebitda": peer_ev_ebitda,
                    "company_ev_ebitda": (market_cap + net_debt) / ebitda,
                    "premium_discount": round((current_price - implied_price) / implied_price * 100, 2),
                }
        
        # Aggregate fair value
        if valuations:
            fair_values = [v["fair_value"] for v in valuations.values()]
            avg_fair_value = np.mean(fair_values)
            upside = (avg_fair_value - current_price) / current_price * 100
            
            return {
                "method": ValuationMethod.PE_RELATIVE.value,
                "fair_value": round(avg_fair_value, 2),
                "current_price": current_price,
                "upside_percent": round(upside, 2),
                "individual_valuations": valuations,
                "confidence": 0.65,
            }
        
        return {"error": "Insufficient data for relative valuation"}

    def earnings_power_value(
        self,
        operating_income: float,
        tax_rate: float,
        wacc: float,
        shares_outstanding: float,
    ) -> Dict[str, Any]:
        """
        Calculate Earnings Power Value (EPV).
        
        Args:
            operating_income: Operating income
            tax_rate: Effective tax rate
            wacc: Weighted average cost of capital
            shares_outstanding: Number of shares outstanding
            
        Returns:
            EPV calculation dictionary
        """
        # Normalize operating income (could add adjustments)
        nopat = operating_income * (1 - tax_rate)
        
        # EPV assumes no growth - value of current earnings perpetuity
        enterprise_value = nopat / wacc
        equity_value = enterprise_value  # Simplified - should subtract net debt
        
        epv_per_share = equity_value / shares_outstanding
        
        return {
            "method": "earnings_power_value",
            "epv_per_share": round(epv_per_share, 2),
            "normalized_nopat": round(nopat, 2),
            "enterprise_value": round(enterprise_value, 2),
            "wacc_used": wacc,
            "assumptions": {
                "growth_assumed": 0,
                "tax_rate": tax_rate,
            },
        }

    def _calculate_wacc(
        self,
        financials: Dict[str, Any],
        beta: float = 1.0,
    ) -> float:
        """Calculate weighted average cost of capital."""
        # Cost of equity using CAPM
        cost_of_equity = self.risk_free_rate + beta * self.equity_risk_premium
        
        # Cost of debt
        interest_expense = financials.get("interest_expense", 0)
        total_debt = financials.get("total_debt", 0)
        tax_rate = financials.get("tax_rate", 0.25)
        
        if total_debt > 0 and interest_expense > 0:
            cost_of_debt = interest_expense / total_debt
        else:
            cost_of_debt = self.risk_free_rate + 0.02  # Default spread
        
        after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
        
        # Capital structure weights
        total_equity = financials.get("total_equity", 0)
        total_capital = total_debt + total_equity
        
        if total_capital > 0:
            weight_debt = total_debt / total_capital
            weight_equity = total_equity / total_capital
        else:
            weight_debt = 0.3  # Default assumption
            weight_equity = 0.7
        
        wacc = weight_equity * cost_of_equity + weight_debt * after_tax_cost_of_debt
        
        return max(0.06, min(0.15, wacc))  # Bound between 6% and 15%

    def _synthesize_fair_value(
        self,
        results: Dict[str, Any],
        current_price: float,
    ) -> Dict[str, Any]:
        """Synthesize fair value estimate from multiple methods."""
        fair_values = []
        methods = []
        
        if "dcf_valuation" in results and "fair_value" in results["dcf_valuation"]:
            fv = results["dcf_valuation"]["fair_value"]
            conf = results["dcf_valuation"].get("confidence", 0.7)
            fair_values.append((fv, conf, "DCF"))
            methods.append(("DCF", fv))
        
        if "relative_valuation" in results and "fair_value" in results["relative_valuation"]:
            fv = results["relative_valuation"]["fair_value"]
            conf = results["relative_valuation"].get("confidence", 0.65)
            fair_values.append((fv, conf, "Relative"))
            methods.append(("Relative", fv))
        
        if "earnings_power" in results and "epv_per_share" in results["earnings_power"]:
            fv = results["earnings_power"]["epv_per_share"]
            conf = 0.60
            fair_values.append((fv, conf, "EPV"))
            methods.append(("EPV", fv))
        
        if not fair_values:
            return {"error": "No valuations available"}
        
        # Weighted average by confidence
        total_weight = sum(conf for _, conf, _ in fair_values)
        weighted_fv = sum(fv * conf for fv, conf, _ in fair_values) / total_weight
        
        # Standard deviation of estimates
        fv_values = [fv for fv, _, _ in fair_values]
        fv_std = np.std(fv_values) if len(fv_values) > 1 else 0
        
        # Confidence based on agreement
        if fv_std / weighted_fv < 0.10:
            overall_confidence = 0.80
        elif fv_std / weighted_fv < 0.20:
            overall_confidence = 0.65
        else:
            overall_confidence = 0.50
        
        upside = (weighted_fv - current_price) / current_price * 100
        
        # Recommendation
        if upside > 25:
            recommendation = "Strong Buy"
        elif upside > 10:
            recommendation = "Buy"
        elif upside > -10:
            recommendation = "Hold"
        elif upside > -25:
            recommendation = "Sell"
        else:
            recommendation = "Strong Sell"
        
        return {
            "fair_value": round(weighted_fv, 2),
            "current_price": current_price,
            "upside_percent": round(upside, 2),
            "confidence": overall_confidence,
            "valuation_range": {
                "low": round(min(fv_values), 2),
                "high": round(max(fv_values), 2),
            },
            "methods_used": methods,
            "recommendation": recommendation,
        }

