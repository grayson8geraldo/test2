"""
Exchange connector using ccxt library.
Supports Bybit futures with testnet option.
"""

import asyncio
from datetime import datetime
from typing import Optional

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd

import config
from bot.utils.logger import log


class ExchangeConnector:
    """Async exchange connector for crypto trading."""

    def __init__(self):
        self.exchange: Optional[ccxt_async.Exchange] = None
        self.exchange_sync: Optional[ccxt.Exchange] = None

    async def connect(self):
        """Initialize exchange connection."""
        exchange_class = getattr(ccxt_async, config.EXCHANGE, None)
        if not exchange_class:
            raise ValueError(f"Exchange {config.EXCHANGE} not supported by ccxt")

        params = {
            "apiKey": config.API_KEY,
            "secret": config.API_SECRET,
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear",
                "adjustForTimeDifference": True,
            },
        }

        if config.USE_TESTNET:
            params["sandbox"] = True

        self.exchange = exchange_class(params)

        # Sync version for non-async contexts
        sync_class = getattr(ccxt, config.EXCHANGE)
        self.exchange_sync = sync_class(params)
        if config.USE_TESTNET:
            self.exchange_sync.set_sandbox_mode(True)

        await self.exchange.load_markets()
        log.info(f"Connected to {config.EXCHANGE} ({'testnet' if config.USE_TESTNET else 'mainnet'})")

    async def close(self):
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "5m", limit: int = 200
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            log.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise

    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker data."""
        try:
            return await self.exchange.fetch_ticker(symbol)
        except Exception as e:
            log.error(f"Error fetching ticker for {symbol}: {e}")
            raise

    async def fetch_balance(self) -> dict:
        """Fetch account balance."""
        try:
            balance = await self.exchange.fetch_balance()
            usdt_free = balance.get("USDT", {}).get("free", 0)
            usdt_total = balance.get("USDT", {}).get("total", 0)
            log.info(f"Balance: {usdt_total:.2f} USDT (free: {usdt_free:.2f})")
            return balance
        except Exception as e:
            log.error(f"Error fetching balance: {e}")
            raise

    async def set_leverage(self, symbol: str, leverage: int):
        """Set leverage for a symbol."""
        try:
            await self.exchange.set_leverage(leverage, symbol)
            log.info(f"Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            log.warning(f"Error setting leverage for {symbol}: {e}")

    async def create_market_order(
        self, symbol: str, side: str, amount: float, params: dict | None = None
    ) -> dict:
        """Place a market order."""
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=amount,
                params=params or {},
            )
            log.info(
                f"Market {side} {amount:.6f} {symbol} "
                f"filled @ {order.get('average', 'N/A')}"
            )
            return order
        except Exception as e:
            log.error(f"Error placing market order: {e}")
            raise

    async def create_limit_order(
        self, symbol: str, side: str, amount: float, price: float, params: dict | None = None
    ) -> dict:
        """Place a limit order."""
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=amount,
                price=price,
                params=params or {},
            )
            log.info(f"Limit {side} {amount:.6f} {symbol} @ {price:.2f}")
            return order
        except Exception as e:
            log.error(f"Error placing limit order: {e}")
            raise

    async def create_stop_loss_order(
        self, symbol: str, side: str, amount: float, stop_price: float
    ) -> dict:
        """Place a stop-loss order."""
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type="stop_market",
                side=side,
                amount=amount,
                params={"stopPrice": stop_price, "closePosition": False},
            )
            log.info(f"Stop-loss {side} {amount:.6f} {symbol} @ {stop_price:.2f}")
            return order
        except Exception as e:
            log.error(f"Error placing stop-loss: {e}")
            raise

    async def create_take_profit_order(
        self, symbol: str, side: str, amount: float, tp_price: float
    ) -> dict:
        """Place a take-profit order."""
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type="take_profit_market",
                side=side,
                amount=amount,
                params={"stopPrice": tp_price, "closePosition": False},
            )
            log.info(f"Take-profit {side} {amount:.6f} {symbol} @ {tp_price:.2f}")
            return order
        except Exception as e:
            log.error(f"Error placing take-profit: {e}")
            raise

    async def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol."""
        try:
            await self.exchange.cancel_all_orders(symbol)
            log.info(f"All orders cancelled for {symbol}")
        except Exception as e:
            log.warning(f"Error cancelling orders for {symbol}: {e}")

    async def fetch_open_positions(self) -> list[dict]:
        """Fetch all open positions."""
        try:
            positions = await self.exchange.fetch_positions()
            open_pos = [
                p for p in positions
                if float(p.get("contracts", 0)) > 0
            ]
            return open_pos
        except Exception as e:
            log.error(f"Error fetching positions: {e}")
            raise

    async def close_position(self, symbol: str, side: str, amount: float) -> dict:
        """Close a position by placing an opposite market order."""
        close_side = "sell" if side == "buy" else "buy"
        return await self.create_market_order(
            symbol, close_side, amount, params={"reduceOnly": True}
        )

    async def fetch_multi_timeframe(
        self, symbol: str, timeframes: dict[str, str], limit: int = 200
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple timeframes concurrently."""
        tasks = {
            name: self.fetch_ohlcv(symbol, tf, limit)
            for name, tf in timeframes.items()
        }
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                log.warning(f"Failed to fetch {name} ({timeframes[name]}) for {symbol}: {e}")
        return results
