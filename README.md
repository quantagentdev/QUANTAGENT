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
  🤖 AI ANALYSIS (Powered by Swarms)
============================================================

📡 Fetching data for BTC-USD...
   ✓ Bitcoin USD - $91163.27
   Market Cap: $1.82T

🌊 MARKET REGIME
   Regime: LOW VOLATILITY
   Confidence: 67%
   Recommended: Breakout Anticipation

📈 TECHNICAL ANALYSIS
   RSI(14): 54.2 - Bullish
   MACD: 598.16 - BULLISH ✅
   Bollinger %B: 76%
   Support: $86,266, $80,659
   Resistance: $94,098

⚠️  RISK METRICS
   Daily VaR (95%): 3.44%
   Sharpe Ratio: -0.07
   Max Drawdown: -32.1%

🎯 PRICE TARGETS
   Target 1: $94,098 ▲ (+3.2%)  - 1-2 weeks
   Target 2: $98,673 ▲ (+8.2%)  - 4-8 weeks
   Target 3: $103,680 ▲ (+13.7%) - 3+ months
   Stop Loss: $85,014

   🟢 BULLISH - Consider long positions
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
