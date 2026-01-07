"""
QuantAgent CLI - Simple command-line interface
"""
import sys
import argparse


def analyze(symbol: str, period: str = "1y", use_ai: bool = False):
    """Run full analysis on a symbol."""
    
    print(f"\n{'='*60}")
    print(f"  $QUANTAGENT - Analyzing {symbol.upper()}")
    print(f"{'='*60}\n")
    
    # Import modules
    from quantagent.data import fetch_data
    from quantagent.technical import TechnicalEngine
    from quantagent.risk import RiskEngine
    from quantagent.regime import MarketRegimeDetector
    from quantagent.targets import PriceTargetGenerator
    from quantagent.signals import SignalAggregator
    import numpy as np
    
    # Fetch real data
    print(f"📡 Fetching data for {symbol}...")
    try:
        data = fetch_data(symbol, period)
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("   Make sure yfinance is installed: pip install yfinance")
        return
    
    info = data["info"]
    current_price = data["current_price"]
    
    print(f"   ✓ {info['name']} - ${current_price:,.2f}")
    if info.get("sector"):
        print(f"   Sector: {info['sector']}")
    if info.get("market_cap"):
        mcap = info["market_cap"]
        if mcap >= 1e12:
            print(f"   Market Cap: ${mcap/1e12:.2f}T")
        elif mcap >= 1e9:
            print(f"   Market Cap: ${mcap/1e9:.2f}B")
        else:
            print(f"   Market Cap: ${mcap/1e6:.2f}M")
    print()
    
    # Calculate returns
    returns = np.diff(data["close"]) / data["close"][:-1]
    
    # 1. Market Regime
    print("🌊 MARKET REGIME")
    print("-" * 40)
    regime_detector = MarketRegimeDetector()
    regime = regime_detector.detect_regime(
        data["close"], data["volume"], data["high"], data["low"]
    )
    print(f"   Regime: {regime.primary_regime.value.upper().replace('_', ' ')}")
    print(f"   Confidence: {regime.confidence:.0%}")
    print(f"   Trend Strength: {regime.trend_strength:.2f}")
    
    strategy = regime_detector.get_optimal_strategy(regime)
    print(f"   Recommended: {strategy['strategy']['approach'].replace('_', ' ').title()}")
    print(f"   Bias: {strategy['strategy']['bias'].upper()}")
    print()
    
    # 2. Technical Analysis
    print("📈 TECHNICAL ANALYSIS")
    print("-" * 40)
    tech = TechnicalEngine()
    
    rsi = tech.rsi(data["close"])
    macd_line, signal_line, histogram = tech.macd(data["close"])
    upper_bb, middle_bb, lower_bb = tech.bollinger_bands(data["close"])
    atr = tech.atr(data["high"], data["low"], data["close"])
    
    rsi_val = rsi[-1]
    macd_val = macd_line[-1]
    signal_val = signal_line[-1]
    atr_val = atr[-1]
    
    # RSI interpretation
    if rsi_val > 70:
        rsi_signal = "OVERBOUGHT ⚠️"
    elif rsi_val < 30:
        rsi_signal = "OVERSOLD 🔥"
    elif rsi_val > 50:
        rsi_signal = "Bullish"
    else:
        rsi_signal = "Bearish"
    
    # MACD interpretation
    if macd_val > signal_val and histogram[-1] > 0:
        macd_signal = "BULLISH ✅"
    elif macd_val < signal_val and histogram[-1] < 0:
        macd_signal = "BEARISH ❌"
    else:
        macd_signal = "Neutral"
    
    # Bollinger position
    bb_position = (current_price - lower_bb[-1]) / (upper_bb[-1] - lower_bb[-1]) * 100 if upper_bb[-1] != lower_bb[-1] else 50
    
    print(f"   RSI(14): {rsi_val:.1f} - {rsi_signal}")
    print(f"   MACD: {macd_val:.2f} - {macd_signal}")
    print(f"   Bollinger %B: {bb_position:.0f}%")
    print(f"   ATR: ${atr_val:.2f} ({atr_val/current_price*100:.1f}%)")
    
    # Support/Resistance
    levels = tech.identify_support_resistance(data["high"], data["low"], data["close"])
    print()
    print(f"   Support: ", end="")
    print(", ".join([f"${s:,.2f}" for s in levels["support"][:3]]) or "None found")
    print(f"   Resistance: ", end="")
    print(", ".join([f"${r:,.2f}" for r in levels["resistance"][:3]]) or "None found")
    print()
    
    # 3. Risk Metrics
    print("⚠️  RISK METRICS")
    print("-" * 40)
    risk = RiskEngine()
    
    var_95 = risk.calculate_var(returns, 0.95)
    cvar_95 = risk.calculate_cvar(returns, 0.95)
    sharpe = risk.calculate_sharpe(returns)
    sortino = risk.calculate_sortino(returns)
    max_dd, _, _ = risk.calculate_max_drawdown(data["close"])
    volatility = np.std(returns) * np.sqrt(252) * 100
    
    print(f"   Daily VaR (95%): {var_95:.2%}")
    print(f"   Daily CVaR (95%): {cvar_95:.2%}")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Sortino Ratio: {sortino:.2f}")
    print(f"   Max Drawdown: {max_dd:.1%}")
    print(f"   Annual Volatility: {volatility:.1f}%")
    print()
    
    # 4. Signal Aggregation
    print("📡 SIGNAL SUMMARY")
    print("-" * 40)
    aggregator = SignalAggregator()
    
    # Add signals based on analysis
    ma_signal = 1 if current_price > middle_bb[-1] else -1
    aggregator.add_technical_signals(
        rsi=rsi_val,
        macd_signal=(macd_val - signal_val) / abs(signal_val) if signal_val != 0 else 0,
        moving_average_signal=ma_signal * 0.5,
    )
    
    signal = aggregator.aggregate()
    
    signal_emoji = {
        "strong_buy": "🟢🟢",
        "buy": "🟢",
        "weak_buy": "🟢",
        "neutral": "⚪",
        "weak_sell": "🔴",
        "sell": "🔴",
        "strong_sell": "🔴🔴",
    }
    
    print(f"   Signal: {signal.signal_type.value.upper().replace('_', ' ')} {signal_emoji.get(signal.signal_type.value, '')}")
    print(f"   Score: {signal.composite_score:.2f}")
    print(f"   Confidence: {signal.confidence:.0%}")
    print()
    
    # 5. Price Targets
    print("🎯 PRICE TARGETS")
    print("-" * 40)
    target_gen = PriceTargetGenerator()
    
    # Determine trend
    trend = "bullish" if regime.trend_strength > 0 and signal.composite_score > 0 else "bearish" if signal.composite_score < -0.2 else "neutral"
    
    targets = target_gen.generate_targets(
        current_price=current_price,
        support_levels=levels["support"],
        resistance_levels=levels["resistance"],
        atr=atr_val,
        trend=trend,
    )
    
    print(f"   Current: ${current_price:,.2f}")
    print()
    
    for i, target in enumerate(targets, 1):
        pct = (target.price - current_price) / current_price * 100
        direction = "▲" if pct > 0 else "▼"
        print(f"   Target {i}: ${target.price:,.2f} {direction} ({pct:+.1f}%)")
        print(f"            Timeframe: {target.timeframe.value.replace('_', ' ')}")
        print(f"            Confidence: {target.confidence:.0%}")
        print(f"            R:R Ratio: {target.risk_reward_ratio:.1f}")
        print()
    
    print(f"   Stop Loss: ${targets[0].invalidation_price:,.2f}")
    print()
    
    # 6. Position Sizing Example
    print("💰 POSITION SIZING (Example: $10,000 account, 2% risk)")
    print("-" * 40)
    position = risk.calculate_position_size(
        account_value=10000,
        entry_price=current_price,
        stop_loss_price=targets[0].invalidation_price,
        target_price=targets[0].price,
        risk_per_trade=0.02,
    )
    
    if "error" not in position:
        print(f"   Shares: {position['shares']:.0f}")
        print(f"   Position Value: ${position['position_value']:,.2f}")
        print(f"   Risk Amount: ${position['risk_amount']:.2f}")
        if position.get('risk_reward_ratio'):
            print(f"   Risk/Reward: {position['risk_reward_ratio']:.2f}")
    print()
    
    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    if signal.signal_type.value in ["strong_buy", "buy"]:
        action = "BULLISH - Consider long positions"
        emoji = "🟢"
    elif signal.signal_type.value in ["strong_sell", "sell"]:
        action = "BEARISH - Consider short positions or exit longs"
        emoji = "🔴"
    else:
        action = "NEUTRAL - Wait for clearer signals"
        emoji = "⚪"
    
    print(f"\n   {emoji} {action}")
    print(f"\n   Key Levels:")
    print(f"   • Entry: ${current_price:,.2f}")
    print(f"   • Target 1: ${targets[0].price:,.2f}")
    print(f"   • Stop Loss: ${targets[0].invalidation_price:,.2f}")
    print()
    
    # 7. AI Analysis (if enabled)
    if use_ai:
        print("=" * 60)
        print("  🤖 AI ANALYSIS (Powered by Swarms)")
        print("=" * 60)
        print()
        
        try:
            from quantagent.agent import QuantAgent
            
            # Prepare data for AI
            ai_data = {
                "symbol": symbol,
                "current_price": current_price,
                "info": info,
                "regime": regime.primary_regime.value,
                "regime_confidence": regime.confidence,
                "trend_strength": regime.trend_strength,
                "rsi": rsi_val,
                "macd": macd_val,
                "macd_signal": signal_val,
                "bb_percent": bb_position,
                "atr": atr_val,
                "atr_percent": atr_val / current_price * 100,
                "support_levels": [f"${s:,.2f}" for s in levels["support"][:3]],
                "resistance_levels": [f"${r:,.2f}" for r in levels["resistance"][:3]],
                "var_95": var_95,
                "cvar_95": cvar_95,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": max_dd,
                "volatility": volatility,
                "signal_type": signal.signal_type.value.upper(),
                "signal_score": signal.composite_score,
                "signal_confidence": signal.confidence,
                "target_1": targets[0].price,
                "target_2": targets[1].price,
                "target_3": targets[2].price,
                "stop_loss": targets[0].invalidation_price,
            }
            
            print("   Generating AI insights...")
            print()
            
            agent = QuantAgent()
            ai_analysis = agent.analyze(ai_data)
            
            print(ai_analysis)
            print()
            
        except ImportError:
            print("   ❌ Swarms not installed. Install with: pip install swarms")
            print("   Then set OPENAI_API_KEY environment variable.")
            print()
        except Exception as e:
            error_msg = str(e).lower()
            if "api_key" in error_msg or "authentication" in error_msg:
                print("   ❌ OpenAI API key not set.")
                print("   Run: export OPENAI_API_KEY='your-key-here'")
            else:
                print(f"   ❌ AI analysis failed: {e}")
            print()
    
    print("⚠️  DISCLAIMER: This is not financial advice. Do your own research.")
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="quantagent",
        description="$QUANTAGENT - Elite Quantitative Trading Analysis",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a stock/crypto")
    analyze_parser.add_argument("symbol", help="Ticker symbol (e.g., AAPL, BTC-USD, ETH-USD)")
    analyze_parser.add_argument("--period", "-p", default="1y", 
                                choices=["1mo", "3mo", "6mo", "1y", "2y"],
                                help="Data period (default: 1y)")
    analyze_parser.add_argument("--ai", action="store_true",
                                help="Enable AI-powered analysis (requires swarms + OpenAI API key)")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        analyze(args.symbol, args.period, use_ai=args.ai)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
