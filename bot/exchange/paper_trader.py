"""
Paper Trading Engine — virtual balance on real market data.

Fetches real prices from Bybit but executes all trades virtually.
Simulates fees, slippage, and position tracking without risking real money.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Optional

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd

import config
from bot.utils.logger import log


class PaperTrader:
    """Paper trading connector — real data, virtual execution."""

    def __init__(self):
        self.exchange: Optional[ccxt_async.Exchange] = None
        self.fee_pct = config.PAPER_FEE_PCT / 100
        self.slippage_pct = config.PAPER_SLIPPAGE_PCT / 100

        # Virtual state
        self.virtual_balance = config.INITIAL_DEPOSIT
        self.virtual_positions: dict[str, dict] = {}  # symbol -> position
        self.virtual_orders: dict[str, list[dict]] = {}  # symbol -> pending orders
        self.order_history: list[dict] = []

        self._state_path = "data/paper_state.json"
        self._load_state()

    async def connect(self):
        """Connect to exchange for real market data only (no auth needed for public endpoints)."""
        exchange_class = getattr(ccxt_async, config.EXCHANGE, None)
        if not exchange_class:
            raise ValueError(f"Exchange {config.EXCHANGE} not supported")

        params = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear",
                "adjustForTimeDifference": True,
            },
        }

        # API keys optional for paper trading (only needed for market data on some endpoints)
        if config.API_KEY:
            params["apiKey"] = config.API_KEY
            params["secret"] = config.API_SECRET

        self.exchange = exchange_class(params)
        await self.exchange.load_markets()

        log.info(
            f"[PAPER] Connected to {config.EXCHANGE} for market data | "
            f"Virtual balance: ${self.virtual_balance:.2f}"
        )

    async def close(self):
        """Close exchange connection and save state."""
        self._save_state()
        if self.exchange:
            await self.exchange.close()

    # ---- Market Data (real) ----

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "5m", limit: int = 200
    ) -> pd.DataFrame:
        """Fetch real OHLCV data from exchange."""
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
        """Fetch real ticker from exchange."""
        try:
            return await self.exchange.fetch_ticker(symbol)
        except Exception as e:
            log.error(f"Error fetching ticker for {symbol}: {e}")
            raise

    async def fetch_balance(self) -> dict:
        """Return virtual balance."""
        return {
            "USDT": {
                "free": self.virtual_balance,
                "total": self.virtual_balance + self._unrealized_pnl(),
                "used": self._margin_in_use(),
            }
        }

    async def fetch_multi_timeframe(
        self, symbol: str, timeframes: dict[str, str], limit: int = 200
    ) -> dict[str, pd.DataFrame]:
        """Fetch real OHLCV for multiple timeframes."""
        results = {}
        for name, tf in timeframes.items():
            try:
                results[name] = await self.fetch_ohlcv(symbol, tf, limit)
            except Exception as e:
                log.warning(f"Failed to fetch {name} for {symbol}: {e}")
        return results

    # ---- Virtual Execution ----

    async def set_leverage(self, symbol: str, leverage: int):
        """Set leverage (tracked virtually)."""
        log.info(f"[PAPER] Leverage set to {leverage}x for {symbol}")

    async def create_market_order(
        self, symbol: str, side: str, amount: float, params: dict | None = None
    ) -> dict:
        """Simulate a market order with slippage and fees."""
        ticker = await self.fetch_ticker(symbol)
        raw_price = ticker["last"]

        # Apply slippage
        if side == "buy":
            fill_price = raw_price * (1 + self.slippage_pct)
        else:
            fill_price = raw_price * (1 - self.slippage_pct)

        # Calculate fee
        notional = amount * fill_price
        fee = notional * self.fee_pct

        # Check if this is a reduce-only (closing) order
        is_reduce = (params or {}).get("reduceOnly", False)

        if not is_reduce:
            # Opening — check margin
            required_margin = notional / config.RISK["leverage"]
            if required_margin > self.virtual_balance:
                raise Exception(
                    f"[PAPER] Insufficient margin: need ${required_margin:.2f}, "
                    f"have ${self.virtual_balance:.2f}"
                )

        order_id = str(uuid.uuid4())[:8]
        order = {
            "id": order_id,
            "symbol": symbol,
            "type": "market",
            "side": side,
            "amount": amount,
            "price": fill_price,
            "average": fill_price,
            "filled": amount,
            "remaining": 0,
            "status": "closed",
            "fee": {"cost": fee, "currency": "USDT"},
            "timestamp": datetime.utcnow().isoformat(),
            "info": {"paper": True},
        }

        self.order_history.append(order)

        log.info(
            f"[PAPER] Market {side} {amount:.6f} {symbol} "
            f"@ {fill_price:.2f} (slippage: {self.slippage_pct*100:.2f}%) "
            f"fee: ${fee:.4f}"
        )

        self._save_state()
        return order

    async def create_stop_loss_order(
        self, symbol: str, side: str, amount: float, stop_price: float
    ) -> dict:
        """Register a virtual stop-loss order."""
        order = {
            "id": str(uuid.uuid4())[:8],
            "symbol": symbol,
            "type": "stop_loss",
            "side": side,
            "amount": amount,
            "stopPrice": stop_price,
            "status": "open",
            "timestamp": datetime.utcnow().isoformat(),
        }

        if symbol not in self.virtual_orders:
            self.virtual_orders[symbol] = []
        self.virtual_orders[symbol].append(order)

        log.info(f"[PAPER] Stop-loss {side} {amount:.6f} {symbol} @ {stop_price:.2f}")
        return order

    async def create_take_profit_order(
        self, symbol: str, side: str, amount: float, tp_price: float
    ) -> dict:
        """Register a virtual take-profit order."""
        order = {
            "id": str(uuid.uuid4())[:8],
            "symbol": symbol,
            "type": "take_profit",
            "side": side,
            "amount": amount,
            "stopPrice": tp_price,
            "status": "open",
            "timestamp": datetime.utcnow().isoformat(),
        }

        if symbol not in self.virtual_orders:
            self.virtual_orders[symbol] = []
        self.virtual_orders[symbol].append(order)

        log.info(f"[PAPER] Take-profit {side} {amount:.6f} {symbol} @ {tp_price:.2f}")
        return order

    async def cancel_all_orders(self, symbol: str):
        """Cancel all virtual pending orders for a symbol."""
        removed = len(self.virtual_orders.get(symbol, []))
        self.virtual_orders[symbol] = []
        if removed:
            log.info(f"[PAPER] Cancelled {removed} orders for {symbol}")

    async def fetch_open_positions(self) -> list[dict]:
        """Return virtual open positions."""
        return [
            {"symbol": sym, **pos}
            for sym, pos in self.virtual_positions.items()
            if pos.get("contracts", 0) > 0
        ]

    async def close_position(self, symbol: str, side: str, amount: float) -> dict:
        """Close a virtual position."""
        close_side = "sell" if side == "buy" else "buy"
        return await self.create_market_order(
            symbol, close_side, amount, params={"reduceOnly": True}
        )

    # ---- Virtual Balance Management ----

    def update_balance(self, pnl: float):
        """Update virtual balance after a trade closes."""
        self.virtual_balance += pnl
        log.info(f"[PAPER] Balance update: {pnl:+.2f} -> ${self.virtual_balance:.2f}")
        self._save_state()

    def get_virtual_balance(self) -> float:
        return self.virtual_balance

    def _margin_in_use(self) -> float:
        """Calculate margin currently locked in positions."""
        margin = 0.0
        for pos in self.virtual_positions.values():
            margin += pos.get("margin", 0)
        return margin

    def _unrealized_pnl(self) -> float:
        """Placeholder for unrealized PnL (calculated by engine)."""
        return 0.0

    # ---- State Persistence ----

    def _save_state(self):
        """Save paper trading state to disk."""
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        state = {
            "virtual_balance": self.virtual_balance,
            "virtual_positions": self.virtual_positions,
            "virtual_orders": self.virtual_orders,
            "order_count": len(self.order_history),
            "last_updated": datetime.utcnow().isoformat(),
        }
        with open(self._state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        """Load paper trading state from disk."""
        if not os.path.exists(self._state_path):
            return

        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self.virtual_balance = state.get("virtual_balance", config.INITIAL_DEPOSIT)
            self.virtual_positions = state.get("virtual_positions", {})
            self.virtual_orders = state.get("virtual_orders", {})
            log.info(
                f"[PAPER] Loaded state: balance=${self.virtual_balance:.2f}, "
                f"positions={len(self.virtual_positions)}"
            )
        except Exception as e:
            log.warning(f"[PAPER] Failed to load state: {e}")
