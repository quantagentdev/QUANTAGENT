"""
QuantAgent - Swarms-Based AI Analysis Agent

Uses the Swarms framework to provide AI-powered trading insights.
"""

from typing import Dict, Any, Optional

# Check if swarms is available
try:
    from swarms import Agent
    SWARMS_AVAILABLE = True
except ImportError:
    SWARMS_AVAILABLE = False


QUANT_SYSTEM_PROMPT = """You are QuantAgent, an elite quantitative trading analyst AI. You receive computed market data and metrics, and your job is to synthesize them into clear, actionable trading insights.

Your analysis style:
- Be direct and confident, but acknowledge uncertainty
- Use specific numbers and levels
- Explain the "why" behind signals
- Highlight key risks
- Give a clear actionable recommendation

When you receive market data, provide:
1. A brief market context (2-3 sentences)
2. Key technical observations (what the indicators are telling us)
3. Risk assessment (what could go wrong)
4. Your trading recommendation with specific entry, targets, and stop loss
5. Confidence level in your analysis

Be concise but thorough. Traders are busy - get to the point."""


class QuantAgent:
    """
    AI-powered quantitative trading agent built on Swarms.
    
    Combines computed technical/risk metrics with LLM reasoning
    to provide synthesized trading insights.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        verbose: bool = False,
    ):
        """
        Initialize the QuantAgent.
        
        Args:
            model_name: LLM model to use
            verbose: Whether to print debug info
        """
        if not SWARMS_AVAILABLE:
            raise ImportError(
                "Swarms is required for AI analysis. Install with: pip install swarms"
            )
        
        self.agent = Agent(
            agent_name="QuantAgent",
            agent_description="Elite Quantitative Trading Analyst",
            system_prompt=QUANT_SYSTEM_PROMPT,
            model_name=model_name,
            max_loops=1,
            verbose=verbose,
            streaming_on=False,
        )
    
    def analyze(self, analysis_data: Dict[str, Any]) -> str:
        """
        Generate AI-powered analysis from computed metrics.
        
        Args:
            analysis_data: Dictionary containing all computed metrics
            
        Returns:
            AI-generated analysis string
        """
        # Format the data for the LLM
        prompt = self._format_analysis_prompt(analysis_data)
        
        # Get AI analysis
        response = self.agent.run(prompt)
        
        return response
    
    def _format_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Format computed metrics into a prompt for the LLM."""
        
        symbol = data.get("symbol", "Unknown")
        current_price = data.get("current_price", 0)
        
        prompt = f"""Analyze this market data for {symbol} and provide your trading recommendation:

## Current Price: ${current_price:,.2f}

## Company/Asset Info
{self._format_dict(data.get("info", {}))}

## Market Regime
- Regime: {data.get("regime", "Unknown")}
- Confidence: {data.get("regime_confidence", 0):.0%}
- Trend Strength: {data.get("trend_strength", 0):.2f}

## Technical Indicators
- RSI(14): {data.get("rsi", 50):.1f}
- MACD: {data.get("macd", 0):.2f} (Signal: {data.get("macd_signal", 0):.2f})
- Bollinger %B: {data.get("bb_percent", 50):.0f}%
- ATR: ${data.get("atr", 0):.2f} ({data.get("atr_percent", 0):.1f}%)

## Key Levels
- Support: {data.get("support_levels", [])}
- Resistance: {data.get("resistance_levels", [])}

## Risk Metrics
- Daily VaR (95%): {data.get("var_95", 0):.2%}
- Daily CVaR (95%): {data.get("cvar_95", 0):.2%}
- Sharpe Ratio: {data.get("sharpe", 0):.2f}
- Sortino Ratio: {data.get("sortino", 0):.2f}
- Max Drawdown: {data.get("max_drawdown", 0):.1%}
- Annual Volatility: {data.get("volatility", 0):.1f}%

## Signal Summary
- Signal: {data.get("signal_type", "NEUTRAL")}
- Score: {data.get("signal_score", 0):.2f}
- Confidence: {data.get("signal_confidence", 0):.0%}

## Computed Price Targets
- Target 1: ${data.get("target_1", 0):,.2f}
- Target 2: ${data.get("target_2", 0):,.2f}
- Target 3: ${data.get("target_3", 0):,.2f}
- Stop Loss: ${data.get("stop_loss", 0):,.2f}

Based on this data, provide your analysis and trading recommendation."""

        return prompt
    
    def _format_dict(self, d: Dict[str, Any]) -> str:
        """Format a dictionary for display."""
        lines = []
        for key, value in d.items():
            if value is not None:
                if isinstance(value, float):
                    if value > 1e9:
                        lines.append(f"- {key}: ${value/1e9:.2f}B")
                    elif value > 1e6:
                        lines.append(f"- {key}: ${value/1e6:.2f}M")
                    else:
                        lines.append(f"- {key}: {value:.2f}")
                else:
                    lines.append(f"- {key}: {value}")
        return "\n".join(lines) if lines else "N/A"

