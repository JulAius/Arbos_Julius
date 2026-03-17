"""
Data Connector - Unified market data fetching via CCXT.

Provides access to BTC price data from multiple exchanges with a simple API.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import ccxt.async_support as ccxt


class DataConnector:
    """
    Unified connector for fetching market data from cryptocurrency exchanges.
    Uses CCXT library for exchange compatibility.
    """

    def __init__(self, exchange_name: str = "binance", api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
        """
        Initialize the data connector.

        Args:
            exchange_name: Name of the exchange (binance, coinbase, hyperliquid, etc.)
            api_key: Optional API key for authenticated endpoints
            api_secret: Optional API secret for authenticated endpoints
        """
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self._exchange: Optional[ccxt.Exchange] = None

    async def __aenter__(self):
        """Async context manager entry."""
        exchange_class = getattr(ccxt, self.exchange_name)
        self._exchange = exchange_class({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
        })
        await self._exchange.load_markets()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._exchange:
            await self._exchange.close()

    async def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            since: Start datetime (if None, fetches last `limit` candles)
            limit: Number of candles to fetch

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if not self._exchange:
            raise RuntimeError("Connector not initialized. Use async with statement.")

        # Convert datetime to timestamp in milliseconds
        since_ts = int(since.timestamp() * 1000) if since else None

        ohlcv = await self._exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since_ts,
            limit=limit
        )

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    async def fetch_multiple_ohlcv(
        self,
        symbols: list,
        timeframe: str = "1h",
        limit: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols in parallel.

        Args:
            symbols: List of trading pair symbols
            timeframe: Candle timeframe
            limit: Number of candles to fetch

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        for symbol in symbols:
            df = await self.fetch_ohlcv(symbol, timeframe, limit=limit)
            results[symbol] = df
        return results


# Convenience function for synchronous usage
def fetch_data_sync(
    exchange_name: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1000
) -> pd.DataFrame:
    """
    Synchronous convenience function to fetch data.

    Example:
        df = fetch_data_sync("binance", "BTC/USDT", "1h", 1000)
    """
    import asyncio

    async def _fetch():
        async with DataConnector(exchange_name) as dc:
            return await dc.fetch_ohlcv(symbol, timeframe, limit=limit)

    return asyncio.run(_fetch())
