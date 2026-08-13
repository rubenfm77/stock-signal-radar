"""
ML signal engine.

Design choices (deliberate, not defaults):
- Target is ATR-normalized forward return over HORIZON_DAYS, not a fixed %.
  This makes the label comparable across a volatile tech stock and a
  low-beta utility, instead of training the model to just chase volatility.
- Validation is walk-forward (expanding window), never a random train/test
  split. Random splits leak future information into training via
  autocorrelated features (indicators computed on rolling windows) and will
  silently inflate accuracy.
- We report precision on the BUY class and a cost-aware backtest, not just
  accuracy, because accuracy is a poor scorecard for signal quality here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from xgboost import XGBClassifier

HORIZON_DAYS = 5
ATR_MOVE_THRESHOLD = 1.0  # forward move must exceed 1x ATR to count as BUY/SELL
FEATURE_COLS = [
    "rsi14", "macd", "macd_hist", "sma50", "sma200",
    "bb_lower", "bb_upper", "atr14", "vote_rsi", "vote_macd",
    "vote_ma_cross", "vote_bollinger", "rules_score",
]
COMMISSION_BPS = 10  # 0.10% per side, adjust to your broker


def build_labels(df: pd.DataFrame, horizon: int = HORIZON_DAYS,
                  atr_threshold: float = ATR_MOVE_THRESHOLD) -> pd.DataFrame:
    """Label = 1 (up), -1 (down), 0 (flat) based on ATR-normalized forward move."""
    out = df.copy()
    fwd_close = out["close"].shift(-horizon)
    fwd_return = fwd_close - out["close"]
    normalized_move = fwd_return / out["atr14"]

    out["fwd_return"] = fwd_return
    out["label"] = 0
    out.loc[normalized_move > atr_threshold, "label"] = 1
    out.loc[normalized_move < -atr_threshold, "label"] = -1
    return out


@dataclass
class WalkForwardResult:
    fold_metrics: list[dict] = field(default_factory=list)
    overall_precision_buy: float = 0.0
    overall_precision_sell: float = 0.0
    backtest_return_pct: float = 0.0
    buy_and_hold_return_pct: float = 0.0
    n_trades: int = 0


def walk_forward_validate(df_labeled: pd.DataFrame, n_folds: int = 5,
                           min_train_size: int = 250) -> WalkForwardResult:
    """
    Expanding-window walk-forward validation.
    Fold k trains on [0:train_end] and tests on the next contiguous block,
    then train_end grows -- the model never sees future data at train time.
    """
    data = df_labeled.dropna(subset=FEATURE_COLS + ["label", "fwd_return"]).copy()
    n = len(data)
    if n < min_train_size + n_folds:
        raise ValueError(f"Not enough data ({n} rows) for {n_folds} walk-forward folds.")

    fold_size = (n - min_train_size) // n_folds
    result = WalkForwardResult()
    all_preds, all_true, all_fwd_returns = [], [], []

    for k in range(n_folds):
        train_end = min_train_size + k * fold_size
        test_end = train_end + fold_size if k < n_folds - 1 else n

        train = data.iloc[:train_end]
        test = data.iloc[train_end:test_end]
        if len(test) == 0:
            continue

        X_train, y_train = train[FEATURE_COLS], train["label"]
        X_test, y_test = test[FEATURE_COLS], test["label"]

        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softmax", num_class=3,
            eval_metric="mlogloss", random_state=42,
        )
        # XGBoost multiclass needs labels in {0,1,2}; remap -1/0/1 -> 0/1/2
        remap = {-1: 0, 0: 1, 1: 2}
        inv_remap = {v: k for k, v in remap.items()}
        model.fit(X_train, y_train.map(remap))
        preds = pd.Series(model.predict(X_test)).map(inv_remap)

        fold_precision_buy = precision_score(y_test, preds, labels=[1], average="micro", zero_division=0)
        fold_precision_sell = precision_score(y_test, preds, labels=[-1], average="micro", zero_division=0)
        result.fold_metrics.append({
            "fold": k + 1,
            "train_rows": len(train),
            "test_rows": len(test),
            "precision_buy": round(fold_precision_buy, 3),
            "precision_sell": round(fold_precision_sell, 3),
        })

        all_preds.extend(preds.tolist())
        all_true.extend(y_test.tolist())
        all_fwd_returns.extend(test["fwd_return"].tolist())

    all_preds = pd.Series(all_preds)
    all_true = pd.Series(all_true)
    result.overall_precision_buy = round(
        precision_score(all_true, all_preds, labels=[1], average="micro", zero_division=0), 3)
    result.overall_precision_sell = round(
        precision_score(all_true, all_preds, labels=[-1], average="micro", zero_division=0), 3)

    # cost-aware backtest: only act on BUY/SELL predictions, apply commission both ways
    fwd = np.array(all_fwd_returns)
    preds_arr = all_preds.to_numpy()
    trade_mask = preds_arr != 0
    result.n_trades = int(trade_mask.sum())
    directional_return = np.where(preds_arr == 1, fwd, np.where(preds_arr == -1, -fwd, 0))
    cost = (COMMISSION_BPS / 10000) * 2  # round trip
    net_returns = np.where(trade_mask, directional_return - cost * np.abs(fwd), 0)
    result.backtest_return_pct = round(float(np.nansum(net_returns)), 2)

    return result
