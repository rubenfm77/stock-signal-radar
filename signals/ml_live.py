"""
Live ML signal for the alert pipeline.

This is deliberately DIFFERENT from signals/ml_model.py's walk_forward_validate:
walk-forward is for *measuring* how trustworthy the model is (backtest,
run manually/occasionally). This module is for *producing* today's signal
cheaply enough to run for ~60 tickers on every scheduled GitHub Action --
it trains once on all available history and predicts the latest row.

If you haven't checked the walk-forward precision for a ticker recently,
treat this live signal with proportional skepticism.
"""
from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier

from signals.ml_model import FEATURE_COLS

REMAP = {-1: 0, 0: 1, 1: 2}
INV_REMAP = {v: k for k, v in REMAP.items()}


def train_and_predict_latest(df_labeled: pd.DataFrame) -> dict | None:
    """
    Train on all rows with a known label (i.e. not in the last HORIZON_DAYS,
    where the forward-looking label can't be computed yet), then predict the
    most recent row (which has no label yet -- that's the point).
    """
    usable = df_labeled.dropna(subset=FEATURE_COLS)
    train_rows = usable.dropna(subset=["label"])
    latest_row = usable.iloc[[-1]]

    if len(train_rows) < 200:
        return None  # not enough history for a meaningful model

    X_train, y_train = train_rows[FEATURE_COLS], train_rows["label"].map(REMAP)
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softmax", num_class=3,
        eval_metric="mlogloss", random_state=42,
    )
    model.fit(X_train, y_train)

    pred = int(pd.Series(model.predict(latest_row[FEATURE_COLS])).map(INV_REMAP).iloc[0])
    label = {1: "BUY", -1: "SELL", 0: "HOLD"}[pred]
    return {"ml_signal": label, "date": latest_row.index[-1]}
