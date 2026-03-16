"""
Self-Learning Module — analyzes performance every 20 closed trades
and adjusts strategy weights and indicator parameters.

What it optimizes:
1. Strategy weights — increase weight for profitable strategies, decrease for losing
2. Indicator sensitivity — adjust RSI/BB/MACD thresholds based on win patterns
3. Risk parameters — tune SL/TP distances based on actual market behavior
4. Pair selection — favor pairs with better win rates
"""

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

import config
from bot.strategies.base import Trade, StrategyName, SignalType
from bot.utils.logger import log


class SelfLearner:

    def __init__(self):
        self.analysis_count = 0
        self.strategy_weights = {
            StrategyName.TREND_FOLLOW.value: config.STRATEGY["trend_follow_weight"],
            StrategyName.MEAN_REVERSION.value: config.STRATEGY["mean_reversion_weight"],
            StrategyName.BREAKOUT.value: config.STRATEGY["breakout_weight"],
            StrategyName.SCALP.value: config.STRATEGY["scalp_weight"],
        }
        self.indicator_adjustments: dict = {}
        self.risk_adjustments: dict = {}
        self.pair_scores: dict[str, float] = {}
        self.analysis_history: list[dict] = []
        self._load_state()

    def should_analyze(self, total_closed_trades: int) -> bool:
        """Check if it's time for a self-analysis cycle."""
        interval = config.LEARNING["analysis_interval"]
        expected_count = total_closed_trades // interval
        return expected_count > self.analysis_count

    def analyze(self, trades: list[Trade]) -> dict:
        """Run full self-analysis on recent trades and adjust parameters.

        Called every 20 closed trades.
        """
        self.analysis_count += 1
        window = config.LEARNING["performance_window"]
        recent = trades[-window:] if len(trades) > window else trades

        log.info(f"=== SELF-ANALYSIS CYCLE #{self.analysis_count} ({len(recent)} trades) ===")

        report = {
            "cycle": self.analysis_count,
            "timestamp": datetime.utcnow().isoformat(),
            "total_trades": len(recent),
        }

        # 1. Analyze by strategy
        strategy_analysis = self._analyze_strategies(recent)
        report["strategy_analysis"] = strategy_analysis

        # 2. Adjust strategy weights
        weight_changes = self._adjust_strategy_weights(strategy_analysis)
        report["weight_changes"] = weight_changes

        # 3. Analyze indicator effectiveness
        indicator_analysis = self._analyze_indicators(recent)
        report["indicator_analysis"] = indicator_analysis

        # 4. Adjust indicator parameters
        indicator_changes = self._adjust_indicators(indicator_analysis, recent)
        report["indicator_changes"] = indicator_changes

        # 5. Analyze risk metrics
        risk_analysis = self._analyze_risk(recent)
        report["risk_analysis"] = risk_analysis

        # 6. Adjust risk parameters
        risk_changes = self._adjust_risk(risk_analysis)
        report["risk_changes"] = risk_changes

        # 7. Score trading pairs
        pair_analysis = self._analyze_pairs(recent)
        report["pair_analysis"] = pair_analysis

        # 8. Generate summary
        report["summary"] = self._generate_summary(report)

        self.analysis_history.append(report)
        self._save_state()

        log.info(f"=== ANALYSIS COMPLETE: {report['summary']} ===")
        return report

    def get_strategy_weight(self, strategy_name: str) -> float:
        return self.strategy_weights.get(strategy_name, 0.25)

    def get_indicator_params(self) -> dict:
        return self.indicator_adjustments

    def get_risk_params(self) -> dict:
        return self.risk_adjustments

    def get_pair_score(self, symbol: str) -> float:
        return self.pair_scores.get(symbol, 1.0)

    # ---- Internal Analysis Methods ----

    def _analyze_strategies(self, trades: list[Trade]) -> dict:
        """Break down performance by strategy."""
        by_strategy = defaultdict(list)
        for t in trades:
            by_strategy[t.strategy.value].append(t)

        analysis = {}
        for name, strades in by_strategy.items():
            wins = [t for t in strades if t.pnl and t.pnl > 0]
            losses = [t for t in strades if t.pnl and t.pnl <= 0]
            total_pnl = sum(t.pnl for t in strades if t.pnl)
            win_rate = len(wins) / len(strades) if strades else 0
            avg_win = np.mean([t.pnl for t in wins]) if wins else 0
            avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0
            profit_factor = sum(t.pnl for t in wins) / sum(abs(t.pnl) for t in losses) if losses else float("inf")

            analysis[name] = {
                "trades": len(strades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "profit_factor": profit_factor,
            }

            log.info(
                f"  Strategy {name}: {len(strades)} trades, "
                f"WR={win_rate:.1%}, PF={profit_factor:.2f}, PnL={total_pnl:+.2f}"
            )

        return analysis

    def _adjust_strategy_weights(self, analysis: dict) -> dict:
        """Adjust strategy weights based on performance."""
        step = config.LEARNING["weight_adjustment_step"]
        changes = {}

        # Calculate performance score for each strategy
        scores = {}
        for name, stats in analysis.items():
            # Score = win_rate * profit_factor (capped)
            pf = min(stats["profit_factor"], 5.0)
            scores[name] = stats["win_rate"] * pf

        if not scores:
            return changes

        avg_score = np.mean(list(scores.values()))

        for name, score in scores.items():
            old_weight = self.strategy_weights.get(name, 0.25)

            if score > avg_score * 1.2:
                # Outperformer — increase weight
                delta = min(step, 0.10)
                changes[name] = f"+{delta:.2f}"
            elif score < avg_score * 0.8:
                # Underperformer — decrease weight
                delta = -min(step, 0.10)
                changes[name] = f"{delta:.2f}"
            else:
                delta = 0
                changes[name] = "no change"

            new_weight = max(0.05, min(0.50, old_weight + delta))
            self.strategy_weights[name] = new_weight

        # Normalize weights to sum to 1
        total = sum(self.strategy_weights.values())
        self.strategy_weights = {k: v / total for k, v in self.strategy_weights.items()}

        log.info(f"  Adjusted weights: {self.strategy_weights}")
        return changes

    def _analyze_indicators(self, trades: list[Trade]) -> dict:
        """Analyze which indicator values were present in winning vs losing trades."""
        winning_indicators = defaultdict(list)
        losing_indicators = defaultdict(list)

        for t in trades:
            target = winning_indicators if (t.pnl and t.pnl > 0) else losing_indicators
            for key, val in t.indicators_at_entry.items():
                if isinstance(val, (int, float)):
                    target[key].append(val)

        analysis = {}
        all_keys = set(list(winning_indicators.keys()) + list(losing_indicators.keys()))
        for key in all_keys:
            w_vals = winning_indicators.get(key, [])
            l_vals = losing_indicators.get(key, [])
            analysis[key] = {
                "win_mean": float(np.mean(w_vals)) if w_vals else None,
                "win_std": float(np.std(w_vals)) if w_vals else None,
                "loss_mean": float(np.mean(l_vals)) if l_vals else None,
                "loss_std": float(np.std(l_vals)) if l_vals else None,
                "win_count": len(w_vals),
                "loss_count": len(l_vals),
            }

        return analysis

    def _adjust_indicators(self, indicator_analysis: dict, trades: list[Trade]) -> dict:
        """Adjust indicator thresholds based on analysis."""
        changes = {}
        step_pct = config.LEARNING["indicator_sensitivity_step"] / 100

        # Adjust RSI thresholds
        if "rsi" in indicator_analysis:
            info = indicator_analysis["rsi"]
            if info["win_mean"] is not None and info["loss_mean"] is not None:
                # If winning trades had lower RSI at entry (shorts), tighten overbought
                rsi_diff = info["win_mean"] - info["loss_mean"]

                current_ob = config.INDICATORS["rsi_overbought"]
                current_os = config.INDICATORS["rsi_oversold"]

                if rsi_diff > 5:
                    # Winning entries had higher RSI → market reverting more from higher
                    new_ob = min(80, current_ob + 2)
                    changes["rsi_overbought"] = f"{current_ob} -> {new_ob}"
                    self.indicator_adjustments["rsi_overbought"] = new_ob
                elif rsi_diff < -5:
                    new_ob = max(60, current_ob - 2)
                    changes["rsi_overbought"] = f"{current_ob} -> {new_ob}"
                    self.indicator_adjustments["rsi_overbought"] = new_ob

        # Adjust BB width sensitivity
        if "bb_pctb" in indicator_analysis:
            info = indicator_analysis["bb_pctb"]
            if info["win_mean"] is not None:
                # Optimal BB %B for entries
                changes["optimal_bb_pctb_entry"] = f"{info['win_mean']:.2f}"

        # Adjust ATR multiplier for stops based on actual outcomes
        winning_atr = [t.indicators_at_entry.get("atr", 0) for t in trades if t.pnl and t.pnl > 0]
        losing_atr = [t.indicators_at_entry.get("atr", 0) for t in trades if t.pnl and t.pnl <= 0]

        if winning_atr and losing_atr:
            avg_win_atr = np.mean(winning_atr)
            avg_lose_atr = np.mean(losing_atr)
            if avg_lose_atr > avg_win_atr * 1.3:
                changes["atr_filter"] = "Avoid high-ATR entries (too volatile)"
                self.indicator_adjustments["max_atr_pct"] = float(avg_win_atr * 1.2)

        log.info(f"  Indicator adjustments: {changes}")
        return changes

    def _analyze_risk(self, trades: list[Trade]) -> dict:
        """Analyze risk metrics from actual trade outcomes."""
        if not trades:
            return {}

        # Actual SL/TP hit rates
        sl_hits = [t for t in trades if t.close_reason == "sl"]
        tp_hits = [t for t in trades if t.close_reason == "tp"]
        trailing_hits = [t for t in trades if t.close_reason == "trailing"]

        # Average actual gain and loss
        winners = [t for t in trades if t.pnl and t.pnl > 0]
        losers = [t for t in trades if t.pnl and t.pnl <= 0]

        avg_actual_gain_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_actual_loss_pct = np.mean([abs(t.pnl_pct) for t in losers]) if losers else 0

        # How many trades hit SL vs TP first
        analysis = {
            "sl_hit_rate": len(sl_hits) / len(trades) if trades else 0,
            "tp_hit_rate": len(tp_hits) / len(trades) if trades else 0,
            "trailing_hit_rate": len(trailing_hits) / len(trades) if trades else 0,
            "avg_actual_gain_pct": float(avg_actual_gain_pct),
            "avg_actual_loss_pct": float(avg_actual_loss_pct),
            "actual_rr": float(avg_actual_gain_pct / avg_actual_loss_pct) if avg_actual_loss_pct > 0 else 0,
        }

        log.info(
            f"  Risk analysis: SL_rate={analysis['sl_hit_rate']:.1%}, "
            f"TP_rate={analysis['tp_hit_rate']:.1%}, "
            f"Actual RR={analysis['actual_rr']:.2f}"
        )

        return analysis

    def _adjust_risk(self, risk_analysis: dict) -> dict:
        """Adjust risk parameters based on analysis."""
        changes = {}

        if not risk_analysis:
            return changes

        # If SL hit rate too high, widen stops
        if risk_analysis.get("sl_hit_rate", 0) > 0.5:
            current_sl = config.RISK["default_sl_pct"]
            new_sl = min(current_sl * 1.15, 3.0)
            changes["default_sl_pct"] = f"{current_sl} -> {new_sl:.2f} (SL hit too often)"
            self.risk_adjustments["default_sl_pct"] = new_sl

        # If TP hit rate too low, tighten TP
        if risk_analysis.get("tp_hit_rate", 0) < 0.2 and risk_analysis.get("sl_hit_rate", 0) < 0.4:
            current_tp = config.RISK["default_tp_pct"]
            new_tp = max(current_tp * 0.85, 1.5)
            changes["default_tp_pct"] = f"{current_tp} -> {new_tp:.2f} (TP rarely hit)"
            self.risk_adjustments["default_tp_pct"] = new_tp

        # If actual RR is good but win rate low, relax SL
        if risk_analysis.get("actual_rr", 0) > 2.0:
            changes["risk_note"] = "Good RR — maintain current stop strategy"

        log.info(f"  Risk adjustments: {changes}")
        return changes

    def _analyze_pairs(self, trades: list[Trade]) -> dict:
        """Score trading pairs by performance."""
        by_pair = defaultdict(list)
        for t in trades:
            by_pair[t.symbol].append(t)

        pair_analysis = {}
        for symbol, ptrades in by_pair.items():
            wins = sum(1 for t in ptrades if t.pnl and t.pnl > 0)
            total_pnl = sum(t.pnl for t in ptrades if t.pnl)
            wr = wins / len(ptrades) if ptrades else 0

            # Score: win_rate * 0.6 + profitability * 0.4
            pnl_score = min(max(total_pnl / 10, -1), 1)  # normalize
            score = wr * 0.6 + (pnl_score + 1) / 2 * 0.4

            self.pair_scores[symbol] = score
            pair_analysis[symbol] = {
                "trades": len(ptrades),
                "win_rate": wr,
                "total_pnl": total_pnl,
                "score": score,
            }

            log.info(f"  Pair {symbol}: {len(ptrades)} trades, WR={wr:.1%}, PnL={total_pnl:+.2f}, Score={score:.2f}")

        return pair_analysis

    def _generate_summary(self, report: dict) -> str:
        """Generate a human-readable summary of the analysis."""
        parts = []
        parts.append(f"Cycle #{report['cycle']}, {report['total_trades']} trades analyzed")

        # Best/worst strategy
        strat = report.get("strategy_analysis", {})
        if strat:
            best = max(strat.items(), key=lambda x: x[1].get("total_pnl", 0))
            worst = min(strat.items(), key=lambda x: x[1].get("total_pnl", 0))
            parts.append(f"Best: {best[0]} (PnL={best[1]['total_pnl']:+.2f})")
            parts.append(f"Worst: {worst[0]} (PnL={worst[1]['total_pnl']:+.2f})")

        # Weight changes
        wc = report.get("weight_changes", {})
        changed = {k: v for k, v in wc.items() if v != "no change"}
        if changed:
            parts.append(f"Weight changes: {changed}")

        return " | ".join(parts)

    def _save_state(self):
        """Persist learning state to disk."""
        if not config.LEARNING["save_learning_data"]:
            return

        state = {
            "analysis_count": self.analysis_count,
            "strategy_weights": self.strategy_weights,
            "indicator_adjustments": self.indicator_adjustments,
            "risk_adjustments": self.risk_adjustments,
            "pair_scores": self.pair_scores,
            "analysis_history": self.analysis_history[-5:],  # keep last 5
        }

        path = config.LEARNING["learning_data_path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        """Load learning state from disk."""
        path = config.LEARNING["learning_data_path"]
        if not os.path.exists(path):
            return

        try:
            with open(path) as f:
                state = json.load(f)

            self.analysis_count = state.get("analysis_count", 0)
            self.strategy_weights = state.get("strategy_weights", self.strategy_weights)
            self.indicator_adjustments = state.get("indicator_adjustments", {})
            self.risk_adjustments = state.get("risk_adjustments", {})
            self.pair_scores = state.get("pair_scores", {})
            self.analysis_history = state.get("analysis_history", [])
            log.info(f"Loaded learning state: {self.analysis_count} analysis cycles completed")
        except Exception as e:
            log.warning(f"Failed to load learning state: {e}")
