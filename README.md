# ⚔️ $QUANTAGENT

**Elite Quantitative Trading Analysis in One Command**

```bash
python -m quantagent analyze BTC-USD --ai
```

That's it. Real data. Full analysis. Price targets. Risk metrics.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/quantagent/quantagent.git
cd quantagent
pip install -r requirements.txt
```

### 2. Analyze Any Stock or Crypto

```bash
# Stocks
python -m quantagent analyze AAPL --ai
python -m quantagent analyze TSLA --ai
python -m quantagent analyze NVDA --ai

# Crypto
python -m quantagent analyze BTC-USD --ai
python -m quantagent analyze ETH-USD --ai
python -m quantagent analyze SOL-USD --ai
```

---

## 🤖 AI-Powered Analysis (Swarms Integration)

Add `--ai` to get AI-powered insights using the **Swarms framework**:

```bash
# First, install swarms and set your API key
pip install swarms
export OPENAI_API_KEY="your-key-here"

# Then run with --ai flag
python -m quantagent analyze BTC-USD --ai
```

This adds an AI analysis section that synthesizes all the computed metrics into actionable trading insights using GPT-4.

```
============================================================
  $QUANTAGENT - Analyzing BTC-USD
============================================================

📡 Fetching data for BTC-USD...
   ✓ Bitcoin USD - $91,052.32
   Market Cap: $1.82T

🌊 MARKET REGIME
----------------------------------------
   Regime: LOW VOLATILITY
   Confidence: 66%
   Trend Strength: 0.06
   Recommended: Breakout Anticipation
   Bias: NEUTRAL

📈 TECHNICAL ANALYSIS
----------------------------------------
   RSI(14): 53.8 - Bullish
   MACD: 589.31 - BULLISH ✅
   Bollinger %B: 75%
   ATR: $2505.83 (2.8%)

   Support: $86,266.74, $80,659.81
   Resistance: $94,098.38

⚠️  RISK METRICS
----------------------------------------
   Daily VaR (95%): 3.44%
   Daily CVaR (95%): 4.83%
   Sharpe Ratio: -0.07
   Sortino Ratio: -0.10
   Max Drawdown: -32.1%
   Annual Volatility: 34.5%

📡 SIGNAL SUMMARY
----------------------------------------
   Signal: BUY 🟢
   Score: 0.52
   Confidence: 60%

🎯 PRICE TARGETS
----------------------------------------
   Current: $91,052.32

   Target 1: $94,098.38 ▲ (+3.3%)
            Timeframe: 1-2 weeks
            Confidence: 70%
            R:R Ratio: 0.5

   Target 2: $98,569.82 ▲ (+8.3%)
            Timeframe: 4-8 weeks
            Confidence: 55%
            R:R Ratio: 1.2

   Target 3: $103,581.49 ▲ (+13.8%)
            Timeframe: 3+ months
            Confidence: 40%
            R:R Ratio: 2.1

   Stop Loss: $85,013.83

💰 POSITION SIZING (Example: $10,000 account, 2% risk)
----------------------------------------
   Shares: 0
   Position Value: $2,000.00
   Risk Amount: $132.64
   Risk/Reward: 0.50

============================================================
  SUMMARY
============================================================

   🟢 BULLISH - Consider long positions

   Key Levels:
   • Entry: $91,052.32
   • Target 1: $94,098.38
   • Stop Loss: $85,013.83

============================================================
  🤖 AI ANALYSIS (Powered by Swarms)
============================================================

   Generating AI insights...

╭──────────────────────────────────────────────────────────────────────────────────────── Agent Name QuantAgent [Max Loops: 1 ] ────────────────────────────────────────────────────────────────────────────────────────╮
│ ### Market Context                                                                                                                                                                                                    │
│ Bitcoin (BTC-USD) is currently trading at $91,052.32, sitting well above its 52-week low of $74,436.68 but significantly below its 52-week high of $126,198.07. The market is in a low-volatility regime, indicating  │
│ a stable price environment, but with a weak trend strength of 0.06, suggesting potential indecision among traders.                                                                                                    │
│                                                                                                                                                                                                                       │
│ ### Key Technical Observations                                                                                                                                                                                        │
│ - **RSI(14)** at 53.8 indicates that Bitcoin is neither overbought nor oversold, allowing for potential upside.                                                                                                       │
│ - **MACD** shows a positive divergence with the MACD line at 589.31 above the signal line at -122.37, signaling bullish momentum.                                                                                     │
│ - **Bollinger %B** at 75% suggests that the price is approaching the upper band, which could indicate a breakout or potential resistance at $94,098.38.                                                               │
│ - **ATR** indicates a relatively low volatility of $2505.83, which may suggest limited price swings in the near term.                                                                                                 │
│                                                                                                                                                                                                                       │
│ ### Risk Assessment                                                                                                                                                                                                   │
│ Key risks include:                                                                                                                                                                                                    │
│ - A potential reversal if Bitcoin fails to break above resistance at $94,098.38, leading to a pullback towards support levels.                                                                                        │
│ - The negative Sharpe and Sortino ratios (-0.07 and -0.10, respectively) indicate that the risk-adjusted returns are unfavorable, which could deter investors.                                                        │
│ - The Daily VaR of 3.44% and CVaR of 4.83% suggest the potential for significant losses in adverse conditions.                                                                                                        │
│                                                                                                                                                                                                                       │
│ ### Trading Recommendation                                                                                                                                                                                            │
│ - **Action**: BUY BTC-USD                                                                                                                                                                                             │
│ - **Entry Point**: $91,052.32                                                                                                                                                                                         │
│ - **Targets**:                                                                                                                                                                                                        │
│   - Target 1: $94,098.38 (immediate resistance)                                                                                                                                                                       │
│   - Target 2: $98,569.82 (next psychological level)                                                                                                                                                                   │
│   - Target 3: $103,581.49 (52-week high proximity)                                                                                                                                                                    │
│ - **Stop Loss**: $85,013.83 (below key support at $86,266.74)                                                                                                                                                         │
│                                                                                                                                                                                                                       │
│ ### Confidence Level                                                                                                                                                                                                  │
│ I have a **confidence level of 60%** in this analysis. The buy signal is supported by technical indicators, but the overall market conditions and risk metrics suggest caution. Monitoring price action around key    │
│ resistance levels will be crucial.                                                                                                                                                                                    │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
### Market Context
Bitcoin (BTC-USD) is currently trading at $91,052.32, sitting well above its 52-week low of $74,436.68 but significantly below its 52-week high of $126,198.07. The market is in a low-volatility regime, indicating a stable price environment, but with a weak trend strength of 0.06, suggesting potential indecision among traders.

### Key Technical Observations
- **RSI(14)** at 53.8 indicates that Bitcoin is neither overbought nor oversold, allowing for potential upside.
- **MACD** shows a positive divergence with the MACD line at 589.31 above the signal line at -122.37, signaling bullish momentum.
- **Bollinger %B** at 75% suggests that the price is approaching the upper band, which could indicate a breakout or potential resistance at $94,098.38.
- **ATR** indicates a relatively low volatility of $2505.83, which may suggest limited price swings in the near term.

### Risk Assessment
Key risks include:
- A potential reversal if Bitcoin fails to break above resistance at $94,098.38, leading to a pullback towards support levels.
- The negative Sharpe and Sortino ratios (-0.07 and -0.10, respectively) indicate that the risk-adjusted returns are unfavorable, which could deter investors.
- The Daily VaR of 3.44% and CVaR of 4.83% suggest the potential for significant losses in adverse conditions.

### Trading Recommendation
- **Action**: BUY BTC-USD
- **Entry Point**: $91,052.32
- **Targets**: 
  - Target 1: $94,098.38 (immediate resistance)
  - Target 2: $98,569.82 (next psychological level)
  - Target 3: $103,581.49 (52-week high proximity)
- **Stop Loss**: $85,013.83 (below key support at $86,266.74)

### Confidence Level
I have a **confidence level of 60%** in this analysis. The buy signal is supported by technical indicators, but the overall market conditions and risk metrics suggest caution. Monitoring price action around key resistance levels will be crucial.

⚠️  DISCLAIMER: This is not financial advice. Do your own research.
```

---

## Options

```bash
# Different time periods
python -m quantagent analyze AAPL --period 6mo
python -m quantagent analyze AAPL -p 3mo

# With AI analysis
python -m quantagent analyze AAPL --ai

# Combine options
python -m quantagent analyze NVDA -p 6mo --ai
```

| Flag | Description |
|------|-------------|
| `--period`, `-p` | Data period: `1mo`, `3mo`, `6mo`, `1y`, `2y` |
| `--ai` | Enable AI-powered analysis (requires swarms + API key) |

---

## What You Get

| Feature | Description |
|---------|-------------|
| **Market Regime** | Bull/Bear/Neutral detection with confidence |
| **Technical Analysis** | RSI, MACD, Bollinger Bands, Support/Resistance |
| **Risk Metrics** | VaR, CVaR, Sharpe Ratio, Max Drawdown |
| **Price Targets** | 3 targets with timeframes and R:R ratios |
| **Position Sizing** | How many shares/coins to buy based on risk |
| **Clear Signal** | BUY / SELL / NEUTRAL with confidence score |
| **AI Insights** | LLM-powered analysis synthesis (with `--ai`) |

---

## Architecture

```
python -m quantagent analyze BTC-USD --ai
                │
                ▼
┌─────────────────────────────────────────────┐
│  1. Fetch Data (yfinance)                   │
│     └── Real OHLCV data from Yahoo Finance  │
│                                             │
│  2. Technical Analysis (technical.py)       │
│     └── RSI, MACD, Bollinger, ATR, S/R      │
│                                             │
│  3. Risk Analysis (risk.py)                 │
│     └── VaR, CVaR, Sharpe, Drawdown         │
│                                             │
│  4. Regime Detection (regime.py)            │
│     └── Bull/Bear/Neutral classification    │
│                                             │
│  5. Signal Aggregation (signals.py)         │
│     └── Combine into BUY/SELL signal        │
│                                             │
│  6. Price Targets (targets.py)              │
│     └── 3 targets + stop loss               │
│                                             │
│  7. AI Analysis (agent.py) [if --ai]        │
│     └── Swarms Agent synthesizes insights   │
└─────────────────────────────────────────────┘
```

---

## Requirements

**Basic (no AI):**
- Python 3.10+
- yfinance, numpy, pandas, scipy

**With AI (`--ai` flag):**
- swarms (`pip install swarms`)
- OpenAI API key (`export OPENAI_API_KEY="..."`)

---

## For Developers

```python
from quantagent import TechnicalEngine, RiskEngine, QuantAgent
from quantagent.data import fetch_data
import numpy as np

# Fetch real data
data = fetch_data("AAPL")

# Technical analysis
tech = TechnicalEngine()
rsi = tech.rsi(data["close"])
print(f"RSI: {rsi[-1]:.1f}")

# Risk analysis
returns = np.diff(data["close"]) / data["close"][:-1]
risk = RiskEngine()
var = risk.calculate_var(returns)
print(f"VaR: {var:.2%}")

# AI analysis (requires swarms)
agent = QuantAgent()
analysis = agent.analyze({"symbol": "AAPL", "rsi": rsi[-1], ...})
print(analysis)
```

---

## About

**$QUANTAGENT** is built on the [Swarms](https://swarms.ai) multi-agent framework. It combines:
- **Quantitative Analysis**: 50+ technical indicators, risk metrics, regime detection
- **AI Synthesis**: LLM-powered insights that explain what the numbers mean

**Swarms Marketplace**: [swarms.ai](https://swarms.ai)

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Disclaimer

⚠️ **This is not financial advice.** QuantAgent provides analysis for educational purposes only. Always do your own research before making investment decisions.

---

<p align="center">
  <b>$QUANTAGENT - One Command. Real Analysis. AI Insights.</b>
</p>
