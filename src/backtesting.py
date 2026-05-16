"""
VN30 Stock Prediction Pipeline — Bước 6: Backtesting

Backtest strategy + benchmarks (Buy & Hold, Random).
"""

import os
import json
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
from loguru import logger

import joblib

from src.data_collection import filename_to_ticker


Trade = namedtuple("Trade", [
    "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
    "return_pct", "pnl", "signal_prob"
])


def select_best_model(results):
    """Chọn model có AUC cao nhất."""
    best_name, best_auc = None, -1

    for h_key, h_results in results.items():
        for m_name, m_data in h_results.items():
            auc = m_data["average"].get("auc", 0)
            if auc > best_auc:
                best_auc = auc
                best_name = m_name
                best_horizon = h_key

    logger.info(f"Best model: {best_name} ({best_horizon}) with AUC={best_auc:.4f}")
    return best_name, best_horizon


def run_backtest(predictions, prices, horizon=1,
                 prob_threshold=0.6, fee_rate=0.0015,
                 position_size=0.1, initial_capital=100_000_000):
    """
    Chạy backtest cho 1 model.

    Strategy:
        T+1: mua cuối hôm nay → bán cuối ngày mai (hold 1 ngày)
        T+3: mua cuối hôm nay → bán cuối ngày +3

    Args:
        predictions: list of fold prediction dicts
        prices: dict[str, DataFrame] — OHLCV per ticker
        horizon: 1 hoặc 3
        prob_threshold: Ngưỡng probability để trade
        fee_rate: Phí mỗi lượt (0.15%)
        position_size: Tỷ lệ vốn mỗi lệnh (10%)
        initial_capital: Vốn ban đầu

    Returns:
        dict: backtest results + equity curve + trade log
    """
    capital = initial_capital
    equity_curve = []
    trades = []

    # Flatten predictions từ tất cả folds
    all_signals = []
    for fold_pred in predictions:
        dates = fold_pred["dates"]
        tickers = fold_pred["tickers"]
        y_proba = fold_pred["y_proba"]

        for i in range(len(dates)):
            if isinstance(dates[i], str):
                dt = pd.Timestamp(dates[i])
            else:
                dt = dates[i]

            prob = float(y_proba[i])
            if prob > prob_threshold:
                all_signals.append({
                    "date": dt,
                    "ticker": tickers[i],
                    "prob": prob,
                })

    # Sort by date
    all_signals.sort(key=lambda x: x["date"])
    logger.info(f"Backtest T+{horizon}: {len(all_signals)} signals "
                f"(threshold={prob_threshold})")

    for signal in all_signals:
        ticker = signal["ticker"]
        signal_date = signal["date"]

        if ticker not in prices:
            continue

        price_df = prices[ticker]
        date_idx = price_df.index

        # Tìm ngày signal trong price data
        if signal_date not in date_idx:
            continue

        signal_pos = date_idx.get_loc(signal_date)

        # Entry: cuối ngày signal (dùng Close)
        entry_idx = signal_pos
        # Exit: cuối ngày + horizon
        exit_idx = signal_pos + horizon

        if exit_idx >= len(date_idx):
            continue

        entry_price = float(price_df["close"].iloc[entry_idx])
        exit_price = float(price_df["close"].iloc[exit_idx])
        entry_date = date_idx[entry_idx]
        exit_date = date_idx[exit_idx]

        # PnL
        trade_amount = capital * position_size
        gross_return = (exit_price / entry_price) - 1
        net_return = gross_return - 2 * fee_rate  # Phí mua + bán
        pnl = trade_amount * net_return

        capital += pnl

        trades.append(Trade(
            ticker=ticker,
            entry_date=str(entry_date.date()),
            exit_date=str(exit_date.date()),
            entry_price=entry_price,
            exit_price=exit_price,
            return_pct=net_return * 100,
            pnl=pnl,
            signal_prob=signal["prob"],
        ))

        equity_curve.append({
            "date": str(exit_date.date()),
            "capital": capital,
        })

    # Compute metrics
    metrics = compute_backtest_metrics(
        trades, equity_curve, initial_capital
    )

    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trades": [t._asdict() for t in trades],
        "n_trades": len(trades),
    }


def run_benchmark_buy_hold(prices, initial_capital=100_000_000):
    """Benchmark: Buy & Hold tất cả mã, equal weight."""
    n_tickers = len(prices)
    if n_tickers == 0:
        return {"total_return": 0, "equity_curve": []}

    alloc_per_ticker = initial_capital / n_tickers
    equity_curve = {}

    for ticker, df in prices.items():
        close = df["close"]
        start_price = close.iloc[0]
        for date, price in close.items():
            date_str = str(date.date())
            if date_str not in equity_curve:
                equity_curve[date_str] = 0
            equity_curve[date_str] += alloc_per_ticker * (price / start_price)

    # Sort and format
    eq_sorted = sorted(equity_curve.items())
    eq_list = [{"date": d, "capital": c} for d, c in eq_sorted]

    final_capital = eq_sorted[-1][1] if eq_sorted else initial_capital
    total_return = (final_capital / initial_capital - 1) * 100

    return {
        "total_return": total_return,
        "final_capital": final_capital,
        "equity_curve": eq_list,
    }


def run_benchmark_random(prices, n_trades, horizon=1, fee_rate=0.0015,
                         position_size=0.1, initial_capital=100_000_000, seed=42):
    """Benchmark: Random entries với cùng frequency."""
    rng = np.random.RandomState(seed)
    capital = initial_capital
    trades = []
    equity_curve = []

    tickers = list(prices.keys())
    all_dates = sorted(set().union(*[set(df.index) for df in prices.values()]))

    for _ in range(n_trades):
        ticker = rng.choice(tickers)
        df = prices[ticker]
        date_idx = df.index

        signal_pos = rng.randint(0, max(1, len(date_idx) - horizon - 1))
        exit_pos = signal_pos + horizon

        if exit_pos >= len(date_idx):
            continue

        entry_price = float(df["close"].iloc[signal_pos])
        exit_price = float(df["close"].iloc[exit_pos])

        trade_amount = capital * position_size
        gross_return = (exit_price / entry_price) - 1
        net_return = gross_return - 2 * fee_rate
        pnl = trade_amount * net_return
        capital += pnl

        equity_curve.append({
            "date": str(date_idx[exit_pos].date()),
            "capital": capital,
        })

    total_return = (capital / initial_capital - 1) * 100

    return {
        "total_return": total_return,
        "final_capital": capital,
        "equity_curve": equity_curve,
        "n_trades": n_trades,
    }


def compute_backtest_metrics(trades, equity_curve, initial_capital):
    """Tính các chỉ số backtest."""
    if not trades:
        return {
            "total_return": 0, "sharpe_ratio": 0,
            "max_drawdown": 0, "win_rate": 0, "calmar_ratio": 0,
        }

    returns = [t.return_pct / 100 for t in trades]
    final_capital = equity_curve[-1]["capital"] if equity_curve else initial_capital

    total_return = (final_capital / initial_capital - 1) * 100

    # Sharpe (annualized, ~252 trading days)
    if len(returns) > 1:
        avg_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    else:
        sharpe = 0

    # Max Drawdown
    capitals = [initial_capital] + [e["capital"] for e in equity_curve]
    peak = capitals[0]
    max_dd = 0
    for c in capitals:
        if c > peak:
            peak = c
        dd = (peak - c) / peak
        if dd > max_dd:
            max_dd = dd

    # Win rate
    wins = sum(1 for r in returns if r > 0)
    win_rate = wins / len(returns) * 100 if returns else 0

    # Calmar ratio
    calmar = (total_return / 100) / max_dd if max_dd > 0 else 0

    return {
        "total_return": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate, 2),
        "calmar_ratio": round(calmar, 4),
        "n_trades": len(trades),
        "final_capital": round(final_capital, 0),
    }


def run_full_backtest(config):
    """Orchestrate toàn bộ backtest."""
    paths = config["paths"]
    bt_cfg = config["backtest"]
    processed_dir = paths["processed_data"]
    metrics_dir = paths["metrics_output"]

    logger.info("Step 6: Backtesting")

    # Load results & predictions
    results_path = os.path.join(metrics_dir, "results.json")
    pred_path = os.path.join(metrics_dir, "predictions.pkl")

    if not os.path.exists(results_path) or not os.path.exists(pred_path):
        raise RuntimeError("Training results not found. Run step 'train' first.")

    with open(results_path) as f:
        results = json.load(f)
    predictions = joblib.load(pred_path)

    # Load prices (OHLCV nằm trong subdirectory ohlcv/)
    prices = {}
    ohlcv_dir = Path(processed_dir) / "ohlcv"
    for f in ohlcv_dir.glob("*.parquet"):
        ticker = filename_to_ticker(f.stem)
        prices[ticker] = pd.read_parquet(f)

    # Select best model
    best_model, best_horizon = select_best_model(results)
    horizon = int(best_horizon.replace("T", ""))

    # Strategy backtest
    best_preds = predictions[best_horizon][best_model]
    strategy = run_backtest(
        best_preds, prices,
        horizon=horizon,
        prob_threshold=bt_cfg["prob_threshold"],
        fee_rate=bt_cfg["fee_rate"],
        position_size=bt_cfg["position_size"],
        initial_capital=bt_cfg["initial_capital"],
    )

    # Benchmarks
    buy_hold = run_benchmark_buy_hold(prices, bt_cfg["initial_capital"])
    random_bt = run_benchmark_random(
        prices, n_trades=strategy["n_trades"],
        horizon=horizon, fee_rate=bt_cfg["fee_rate"],
        position_size=bt_cfg["position_size"],
        initial_capital=bt_cfg["initial_capital"],
    )

    backtest_results = {
        "best_model": best_model,
        "best_horizon": best_horizon,
        "strategy": strategy,
        "buy_hold": buy_hold,
        "random": random_bt,
    }

    # Save
    bt_path = os.path.join(metrics_dir, "backtest_results.json")
    with open(bt_path, "w") as f:
        json.dump(backtest_results, f, indent=2, default=str)

    logger.info("\nBACKTEST RESULTS:")
    logger.info(f"  Best model: {best_model} ({best_horizon})")
    s = strategy["metrics"]
    logger.info(f"  Strategy:   Return={s['total_return']:.2f}%, "
                f"Sharpe={s['sharpe_ratio']:.4f}, "
                f"MaxDD={s['max_drawdown']:.2f}%, "
                f"WinRate={s['win_rate']:.1f}%")
    logger.info(f"  Buy & Hold: Return={buy_hold['total_return']:.2f}%")
    logger.info(f"  Random:     Return={random_bt['total_return']:.2f}%")

    return backtest_results
