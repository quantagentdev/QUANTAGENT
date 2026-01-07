"""
QuantAgent Tests
"""
import pytest
import numpy as np


class TestTechnicalEngine:
    """Test technical analysis."""
    
    def test_rsi(self):
        from quantagent import TechnicalEngine
        
        engine = TechnicalEngine()
        prices = np.cumsum(np.random.randn(50)) + 100
        rsi = engine.rsi(prices)
        
        assert len(rsi) == len(prices)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert all(0 <= r <= 100 for r in valid_rsi)
    
    def test_macd(self):
        from quantagent import TechnicalEngine
        
        engine = TechnicalEngine()
        prices = np.cumsum(np.random.randn(50)) + 100
        macd, signal, hist = engine.macd(prices)
        
        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
    
    def test_bollinger(self):
        from quantagent import TechnicalEngine
        
        engine = TechnicalEngine()
        prices = np.cumsum(np.random.randn(50)) + 100
        upper, middle, lower = engine.bollinger_bands(prices)
        
        assert len(upper) == len(prices)


class TestRiskEngine:
    """Test risk analysis."""
    
    def test_var(self):
        from quantagent import RiskEngine
        
        engine = RiskEngine()
        returns = np.random.randn(100) * 0.02
        var = engine.calculate_var(returns)
        
        assert var > 0
    
    def test_sharpe(self):
        from quantagent import RiskEngine
        
        engine = RiskEngine()
        returns = np.random.randn(100) * 0.02
        sharpe = engine.calculate_sharpe(returns)
        
        assert not np.isnan(sharpe)
    
    def test_position_size(self):
        from quantagent import RiskEngine
        
        engine = RiskEngine()
        result = engine.calculate_position_size(
            account_value=10000,
            entry_price=100,
            stop_loss_price=95,
            risk_per_trade=0.02
        )
        
        assert "shares" in result
        assert result["shares"] > 0


class TestRegimeDetector:
    """Test regime detection."""
    
    def test_detect_regime(self):
        from quantagent import MarketRegimeDetector
        
        detector = MarketRegimeDetector()
        prices = np.cumsum(np.random.randn(100)) + 100
        volume = np.random.randint(1000, 10000, 100).astype(float)
        
        regime = detector.detect_regime(prices, volume)
        
        assert regime.primary_regime is not None
        assert 0 <= regime.confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

