"""
Data Fetching Module - Get real market data
"""
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def fetch_data(symbol: str, period: str = "1y") -> Dict[str, Any]:
    """
    Fetch real market data for a symbol.
    
    Args:
        symbol: Stock/crypto ticker (e.g., 'AAPL', 'BTC-USD', 'ETH-USD')
        period: Data period ('1mo', '3mo', '6mo', '1y', '2y')
    
    Returns:
        Dictionary with OHLCV data and info
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required. Install with: pip install yfinance")
    
    ticker = yf.Ticker(symbol)
    
    # Get historical data
    hist = ticker.history(period=period)
    
    if hist.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    
    # Get ticker info
    try:
        info = ticker.info
    except:
        info = {}
    
    return {
        "symbol": symbol,
        "open": hist["Open"].values,
        "high": hist["High"].values,
        "low": hist["Low"].values,
        "close": hist["Close"].values,
        "volume": hist["Volume"].values.astype(float),
        "dates": hist.index.tolist(),
        "current_price": hist["Close"].values[-1],
        "info": {
            "name": info.get("shortName", symbol),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
            "beta": info.get("beta"),
        }
    }

