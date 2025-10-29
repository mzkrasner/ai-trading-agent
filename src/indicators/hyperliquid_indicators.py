"""Calculate technical indicators using Hyperliquid's native OHLCV data."""

import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta

class HyperliquidIndicators:
    """Calculate technical indicators from Hyperliquid candle data."""
    
    def __init__(self, hyperliquid_api):
        """Initialize with a HyperliquidAPI instance."""
        self.hl = hyperliquid_api
        self._candle_cache = {}  # Cache to avoid repeated API calls
        
    async def get_candles(self, coin: str, interval: str, num_candles: int = 100) -> List[Dict]:
        """Fetch OHLCV candles from Hyperliquid.
        
        Args:
            coin: Asset symbol (e.g., "BTC")
            interval: Time interval (e.g., "5m", "4h")
            num_candles: Number of candles to fetch
            
        Returns:
            List of candle dictionaries with OHLCV data
        """
        try:
            # Add delay to avoid rate limiting (30 calls should take ~15 seconds total)
            await asyncio.sleep(0.5)  # 500ms delay between requests to stay well under burst limits
            
            # Calculate time range
            end_time = int(datetime.now().timestamp() * 1000)
            
            # Calculate start time based on interval and number of candles
            interval_ms = self._interval_to_ms(interval)
            start_time = end_time - (interval_ms * num_candles)
            
            # Use the Hyperliquid info client to fetch candles
            candles = await self.hl._retry(
                lambda: self.hl.info.candles_snapshot(
                    name=coin,  # SDK uses 'name' not 'coin'
                    interval=interval,
                    startTime=start_time,
                    endTime=end_time
                ),
                to_thread=True
            )
            
            return candles if candles else []
        except Exception as e:
            logging.error(f"Error fetching candles for {coin} {interval}: {e}")
            return []
    
    def _interval_to_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds."""
        mapping = {
            "1m": 60000,
            "3m": 180000,
            "5m": 300000,
            "15m": 900000,
            "30m": 1800000,
            "1h": 3600000,
            "2h": 7200000,
            "4h": 14400000,
            "8h": 28800000,
            "12h": 43200000,
            "1d": 86400000,
            "3d": 259200000,
            "1w": 604800000,
        }
        return mapping.get(interval, 300000)  # Default to 5m
    
    def calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return None
        
        # Use numpy for efficient calculation
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        
        # Calculate weighted average for the most recent 'period' prices
        recent_prices = prices[-period:]
        ema = np.dot(recent_prices, weights)
        
        return round(float(ema), 4)
    
    def calculate_sma(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return None
        
        return round(sum(prices[-period:]) / period, 4)
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes
        deltas = np.diff(prices[-period-1:])
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0  # Keep only positive changes
        losses[losses > 0] = 0  # Keep only negative changes
        losses = np.abs(losses)  # Convert losses to positive values
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        # Handle edge cases to avoid division by zero
        if avg_loss == 0:
            if avg_gain == 0:
                return 50.0  # Neutral RSI when no price movement
            return 100.0  # Max RSI when only gains
        
        if avg_gain == 0:
            return 0.0  # Min RSI when only losses
        
        # Calculate RSI (only when avg_loss > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Handle any NaN or Infinity values
        if np.isnan(rsi) or np.isinf(rsi):
            return 50.0  # Return neutral RSI if calculation fails
        
        return round(float(rsi), 2)
    
    def calculate_macd(self, prices: List[float], 
                      fast_period: int = 12, 
                      slow_period: int = 26, 
                      signal_period: int = 9) -> Dict[str, Optional[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        
        if len(prices) < slow_period:
            return {"macd": None, "signal": None, "histogram": None}
        
        # Calculate EMAs
        ema_fast = self._calculate_ema_series(prices, fast_period)
        ema_slow = self._calculate_ema_series(prices, slow_period)
        
        if not ema_fast or not ema_slow:
            return {"macd": None, "signal": None, "histogram": None}
        
        # Calculate MACD line
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        
        # Calculate signal line (EMA of MACD)
        if len(macd_line) < signal_period:
            return {"macd": round(macd_line[-1], 4), "signal": None, "histogram": None}
        
        signal_line = self._calculate_ema_series(macd_line, signal_period)
        if not signal_line:
            return {"macd": round(macd_line[-1], 4), "signal": None, "histogram": None}
        
        # Calculate histogram
        histogram = macd_line[-1] - signal_line[-1]
        
        return {
            "macd": round(macd_line[-1], 4),
            "signal": round(signal_line[-1], 4),
            "histogram": round(histogram, 4)
        }
    
    def _calculate_ema_series(self, data: List[float], period: int) -> List[float]:
        """Calculate EMA for entire series (helper for MACD)."""
        if len(data) < period:
            return []
        
        ema = []
        multiplier = 2 / (period + 1)
        
        # Start with SMA for first EMA value
        sma = sum(data[:period]) / period
        ema.append(sma)
        
        # Calculate EMA for rest of the data
        for price in data[period:]:
            ema_val = (price - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_val)
        
        return ema
    
    def calculate_atr(self, candles: List[Dict], period: int = 14) -> Optional[float]:
        """Calculate Average True Range."""
        if len(candles) < period + 1:
            return None
        
        true_ranges = []
        for i in range(1, len(candles)):
            high = float(candles[i]['h'])
            low = float(candles[i]['l'])
            prev_close = float(candles[i-1]['c'])
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            return None
        
        # Calculate ATR as SMA of true ranges
        atr = sum(true_ranges[-period:]) / period
        
        return round(atr, 2)
    
    def calculate_vwap(self, candles: List[Dict], lookback: Optional[int] = None) -> Optional[float]:
        """Calculate Volume-Weighted Average Price from candles.
        
        Args:
            candles: List of candle dicts with 'h', 'l', 'c', 'v' fields
            lookback: Number of candles to use (None = all candles)
            
        Returns:
            VWAP value or None if insufficient data
        """
        try:
            if not candles or len(candles) < 2:
                return None
            
            # Use most recent N candles
            candles_to_use = candles[-lookback:] if lookback else candles
            
            total_pv = 0.0  # price × volume
            total_volume = 0.0
            
            for candle in candles_to_use:
                try:
                    high = float(candle.get('h', 0))
                    low = float(candle.get('l', 0))
                    close = float(candle.get('c', 0))
                    volume = float(candle.get('v', 0))
                    
                    if volume <= 0:
                        continue
                    
                    # Typical price = (H + L + C) / 3
                    typical_price = (high + low + close) / 3
                    
                    total_pv += typical_price * volume
                    total_volume += volume
                except (ValueError, TypeError, KeyError):
                    continue
            
            if total_volume == 0:
                return None
            
            vwap = total_pv / total_volume
            return round(vwap, 2)
            
        except Exception:
            return None
    
    async def get_indicators(self, coin: str, interval: str) -> Dict:
        """Get all key indicators for a coin and interval.
        
        Args:
            coin: Asset symbol (e.g., "BTC")
            interval: Time interval (e.g., "5m", "4h")
            
        Returns:
            Dictionary with calculated indicators
        """
        # Fetch candles (need extra for calculations)
        candles = await self.get_candles(coin, interval, num_candles=100)
        return await self.get_indicators_from_candles(coin, interval, candles)
    
    async def get_indicators_from_candles(self, coin: str, interval: str, candles: List[Dict]) -> Dict:
        """Get all key indicators from pre-fetched candles.
        
        Args:
            coin: Asset symbol (e.g., "BTC")
            interval: Time interval (e.g., "5m", "4h")
            candles: Pre-fetched candle data
            
        Returns:
            Dictionary with calculated indicators
        """
        if not candles:
            logging.error(f"No candles received for {coin} {interval}")
            return {}
        
        # Extract close prices
        closes = [float(c['c']) for c in candles]
        
        # Calculate indicators
        indicators = {
            "ema20": self.calculate_ema(closes, 20),
            "ema50": self.calculate_ema(closes, 50),
            "sma20": self.calculate_sma(closes, 20),
            "rsi7": self.calculate_rsi(closes, 7),
            "rsi14": self.calculate_rsi(closes, 14),
            "atr14": self.calculate_atr(candles, 14),
        }
        
        # Add MACD
        macd_data = self.calculate_macd(closes)
        indicators["macd"] = macd_data.get("macd")
        indicators["macd_signal"] = macd_data.get("signal")
        indicators["macd_histogram"] = macd_data.get("histogram")
        
        # Sanitize all values to ensure JSON compatibility
        for key, value in indicators.items():
            if value is not None and isinstance(value, (float, np.float64, np.float32)):
                if np.isnan(value) or np.isinf(value):
                    indicators[key] = None  # Replace NaN/Inf with None
        
        return indicators
    
    def get_series(self, candles: List[Dict], indicator: str, period: int = None) -> List[float]:
        """Get a series of indicator values from candles.
        
        Args:
            candles: List of OHLCV candles
            indicator: Indicator name (e.g., "rsi", "ema")
            period: Period for the indicator
            
        Returns:
            List of indicator values
        """
        closes = [float(c['c']) for c in candles]
        series = []
        
        # Calculate indicator for each point where we have enough data
        if indicator == "rsi":
            period = period or 14
            for i in range(period + 1, len(closes) + 1):
                value = self.calculate_rsi(closes[:i], period)
                # Ensure no Infinity or NaN values in the series
                if value is None or np.isnan(value) or np.isinf(value):
                    value = 50.0  # Neutral RSI
                series.append(value)
                
        elif indicator == "ema":
            period = period or 20
            for i in range(period, len(closes) + 1):
                value = self.calculate_ema(closes[:i], period)
                series.append(value if value else 0)
                
        elif indicator == "macd":
            for i in range(26, len(closes) + 1):  # Need at least 26 for MACD
                macd_data = self.calculate_macd(closes[:i])
                series.append(macd_data.get("macd", 0))
        
        # Return last 10 values (or all if less than 10)
        return series[-10:] if len(series) > 10 else series
    
    def calculate_market_metrics(self, candles: List[Dict], indicators: Dict) -> Dict:
        """Calculate objective market metrics without prescriptive interpretation.
        
        Args:
            candles: List of OHLCV candles
            indicators: Pre-calculated indicators dictionary
            
        Returns:
            Dictionary of market metrics
        """
        if len(candles) < 20:
            return {}
        
        metrics = {}
        
        # Extract price data
        closes = [float(c['c']) for c in candles]
        highs = [float(c['h']) for c in candles]
        lows = [float(c['l']) for c in candles]
        volumes = [float(c['v']) for c in candles]
        
        # Current values
        current_price = closes[-1]
        
        # Trend Metrics
        if indicators.get('ema20') and indicators.get('ema50'):
            # EMA relationship (positive = bullish structure, negative = bearish)
            ema_separation = (indicators['ema20'] - indicators['ema50']) / indicators['ema50']
            metrics['ema_separation_ratio'] = round(ema_separation, 5)
            
            # Price position relative to EMAs
            price_vs_ema20 = (current_price - indicators['ema20']) / indicators['ema20']
            metrics['price_ema20_deviation'] = round(price_vs_ema20, 5)
            
            price_vs_ema50 = (current_price - indicators['ema50']) / indicators['ema50']
            metrics['price_ema50_deviation'] = round(price_vs_ema50, 5)
        
        # Volatility Metrics
        # Calculate normalized ranges (high-low as % of close)
        recent_ranges = [(highs[i] - lows[i]) / closes[i] for i in range(-20, 0)]
        current_range = recent_ranges[-1]
        avg_range_20 = sum(recent_ranges) / len(recent_ranges)
        
        metrics['current_range_percent'] = round(current_range, 5)
        metrics['avg_range_percent_20'] = round(avg_range_20, 5)
        metrics['volatility_ratio'] = round(current_range / avg_range_20 if avg_range_20 > 0 else 1, 3)
        
        # ATR-based volatility
        if indicators.get('atr14'):
            metrics['atr_price_ratio'] = round(indicators['atr14'] / current_price, 5)
        
        # Momentum Metrics
        if indicators.get('rsi14'):
            metrics['rsi_value'] = indicators['rsi14']
            metrics['rsi_distance_from_50'] = round(indicators['rsi14'] - 50, 2)
            
        if indicators.get('macd') and indicators.get('macd_signal'):
            metrics['macd_value'] = indicators['macd']
            metrics['macd_signal_value'] = indicators['macd_signal']
            metrics['macd_cross_distance'] = round(indicators['macd'] - indicators['macd_signal'], 5)
            
        if indicators.get('macd_histogram'):
            metrics['macd_histogram'] = indicators['macd_histogram']
            # Calculate histogram slope (change over last 3 candles if available)
            if len(candles) >= 3:
                hist_3_back = self.calculate_macd(closes[-3:])
                if hist_3_back and hist_3_back.get('histogram'):
                    hist_change = indicators['macd_histogram'] - hist_3_back['histogram']
                    metrics['macd_histogram_slope_3'] = round(hist_change, 5)
        
        # Market Structure Metrics
        # Count higher highs and lower lows
        metrics['higher_highs_count_20'] = self._count_higher_highs(highs[-20:])
        metrics['lower_lows_count_20'] = self._count_lower_lows(lows[-20:])
        metrics['higher_lows_count_20'] = self._count_higher_lows(lows[-20:])
        metrics['lower_highs_count_20'] = self._count_lower_highs(highs[-20:])
        
        # Consecutive candle analysis
        metrics['consecutive_green_candles'] = self._count_consecutive_green(candles)
        metrics['consecutive_red_candles'] = self._count_consecutive_red(candles)
        
        # Price position within recent range
        recent_high = max(highs[-20:])
        recent_low = min(lows[-20:])
        if recent_high > recent_low:
            range_position = (current_price - recent_low) / (recent_high - recent_low)
            metrics['range_position_20'] = round(range_position, 3)
            metrics['distance_from_high_20'] = round((recent_high - current_price) / current_price, 5)
            metrics['distance_from_low_20'] = round((current_price - recent_low) / current_price, 5)
        
        # Volume Metrics
        if len(volumes) > 20:
            current_volume = volumes[-1]
            avg_volume_20 = sum(volumes[-20:]) / 20
            metrics['volume_ratio_20'] = round(current_volume / avg_volume_20 if avg_volume_20 > 0 else 1, 3)
            metrics['volume_trend_5'] = round(sum(volumes[-5:]) / sum(volumes[-10:-5]) if sum(volumes[-10:-5]) > 0 else 1, 3)
        
        # Price Action Metrics
        # Measure recent price velocity
        if len(closes) >= 10:
            price_change_5 = (closes[-1] - closes[-6]) / closes[-6]
            price_change_10 = (closes[-1] - closes[-11]) / closes[-11]
            metrics['price_velocity_5'] = round(price_change_5, 5)
            metrics['price_velocity_10'] = round(price_change_10, 5)
            
        # Candle body vs wick analysis (current candle)
        current_candle = candles[-1]
        open_price = float(current_candle['o'])
        high = float(current_candle['h'])
        low = float(current_candle['l'])
        close = float(current_candle['c'])
        
        body_size = abs(close - open_price)
        candle_range = high - low
        if candle_range > 0:
            body_ratio = body_size / candle_range
            metrics['current_body_ratio'] = round(body_ratio, 3)
            
            # Upper and lower wick ratios
            upper_wick = high - max(open_price, close)
            lower_wick = min(open_price, close) - low
            metrics['upper_wick_ratio'] = round(upper_wick / candle_range, 3)
            metrics['lower_wick_ratio'] = round(lower_wick / candle_range, 3)
        
        return metrics
    
    def _count_higher_highs(self, highs: List[float]) -> int:
        """Count number of higher highs in sequence."""
        count = 0
        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                count += 1
        return count
    
    def _count_lower_lows(self, lows: List[float]) -> int:
        """Count number of lower lows in sequence."""
        count = 0
        for i in range(1, len(lows)):
            if lows[i] < lows[i-1]:
                count += 1
        return count
    
    def _count_higher_lows(self, lows: List[float]) -> int:
        """Count number of higher lows in sequence."""
        count = 0
        for i in range(1, len(lows)):
            if lows[i] > lows[i-1]:
                count += 1
        return count
    
    def _count_lower_highs(self, highs: List[float]) -> int:
        """Count number of lower highs in sequence."""
        count = 0
        for i in range(1, len(highs)):
            if highs[i] < highs[i-1]:
                count += 1
        return count
    
    def _count_consecutive_green(self, candles: List[Dict]) -> int:
        """Count consecutive green candles from most recent."""
        count = 0
        for candle in reversed(candles):
            if float(candle['c']) > float(candle['o']):
                count += 1
            else:
                break
        return count
    
    def _count_consecutive_red(self, candles: List[Dict]) -> int:
        """Count consecutive red candles from most recent."""
        count = 0
        for candle in reversed(candles):
            if float(candle['c']) < float(candle['o']):
                count += 1
            else:
                break
        return count
