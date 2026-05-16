"""
VN30 — Bước 4c: Stacking Ensemble Training

Theo CLAUDE.md §4c:
  - Meta-learner: Logistic Regression (tránh overfit)
  - Input: OOF probabilities từ XGB + LGBM + LSTM (3 features per horizon)
  - Train trên toàn bộ OOF, validate trên holdout 20% cuối
  - Chỉ train sau khi có đủ OOF từ 4a và 4b
"""

import os, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import joblib
from loguru import logger


def _load_oof(metrics_dir: Path, filename: str) -> dict | None:
    path = metrics_dir / filename
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _align_oof_predictions(oof_xgb_lgbm: dict, oof_lstm: dict, horizon_key: str):
    """
    Align OOF predictions từ XGB, LGBM, LSTM theo (date, ticker) pairs.

    XGB + LGBM là bắt buộc. LSTM là optional:
      - Nếu LSTM có OOF, dùng 3 meta-features (xgb, lgbm, lstm).
      - Nếu LSTM thiếu cho một số (date, ticker) pairs, fill lstm_proba = 0.5
        (neutral) thay vì bỏ samples → giữ data coverage.
      - Nếu LSTM hoàn toàn không có, dùng 2 meta-features.

    Returns: (X_meta [N, 2 or 3], y_true [N])
    """
    # Collect từ XGB+LGBM OOF
    xgb_preds = oof_xgb_lgbm.get(horizon_key, {}).get("xgboost", [])
    lgbm_preds = oof_xgb_lgbm.get(horizon_key, {}).get("lightgbm", [])
    lstm_preds = oof_lstm.get(horizon_key, {}).get("lstm", []) if oof_lstm else []

    def _flatten(fold_list):
        records = {}
        n_overwritten = 0
        for fold in fold_list:
            for d, t, yt, yp in zip(fold["dates"], fold["tickers"],
                                     fold["y_true"], fold["y_proba"]):
                # Chuẩn hóa date string thành ISO format YYYY-MM-DD để tránh
                # string-sort lỗi khi date format không zero-padded.
                try:
                    d_str = pd.Timestamp(d).strftime("%Y-%m-%d")
                except Exception:
                    d_str = str(d)
                key = (d_str, t)
                if key in records:
                    n_overwritten += 1
                records[key] = {"y_true": yt, "y_proba": yp}
        if n_overwritten > 0:
            logger.warning(
                f"  OOF flatten: {n_overwritten} (date, ticker) duplicates detected "
                f"— bị overwrite bởi fold sau. Kiểm tra step_days < test_days trong config."
            )
        return records

    xgb_map = _flatten(xgb_preds)
    lgbm_map = _flatten(lgbm_preds)
    lstm_map = _flatten(lstm_preds)

    # Bắt buộc: XGB + LGBM đều phải có
    common = set(xgb_map.keys()) & set(lgbm_map.keys())
    has_lstm = bool(lstm_map)

    if not common:
        logger.warning(f"No common OOF predictions for {horizon_key}")
        return None, None

    # Sort theo (date_ISO, ticker) — đảm bảo chronological order vì date đã chuẩn hóa
    # YYYY-MM-DD ở bước _flatten, nên string sort == chronological sort.
    common = sorted(common)
    X_meta = []
    y_true = []

    for key in common:
        row = [xgb_map[key]["y_proba"], lgbm_map[key]["y_proba"]]
        if has_lstm:
            # Fill 0.5 (neutral) nếu LSTM thiếu cho sample này — giữ data coverage
            row.append(lstm_map[key]["y_proba"] if key in lstm_map else 0.5)
        X_meta.append(row)
        y_true.append(xgb_map[key]["y_true"])

    if has_lstm:
        n_filled = sum(1 for k in common if k not in lstm_map)
        if n_filled > 0:
            logger.info(f"  LSTM missing for {n_filled}/{len(common)} samples — filled with 0.5")

    return np.array(X_meta, dtype=np.float32), np.array(y_true, dtype=np.int32)


def run_ensemble_training(config):
    """
    Train Stacking Ensemble — Logistic Regression meta-learner.
    Chỉ chạy sau khi có OOF từ XGB/LGBM (bắt buộc) và LSTM (optional).

    Validation: Holdout 20% cuối (chronological) thay vì Walk-Forward.
    Rationale: Meta-learner LogReg chỉ có 2-3 features, Walk-Forward sẽ tạo
    quá nhiều folds nhỏ gây overfit. Holdout đơn giản và ổn định hơn cho
    low-dimensional meta-learner.
    """
    metrics_dir = Path(config["paths"]["metrics_output"])
    models_dir = Path(config["paths"]["models_output"])
    models_dir.mkdir(parents=True, exist_ok=True)
    horizons = config["labels"]["horizons"]

    # Load OOF
    oof_xgb_lgbm = _load_oof(metrics_dir, "xgb_lgbm_oof.pkl")
    if not oof_xgb_lgbm:
        logger.error("XGB/LGBM OOF not found. Run train_xgb_lgbm.py first.")
        return None

    oof_lstm = _load_oof(metrics_dir, "lstm_oof.pkl")
    if not oof_lstm:
        logger.warning("LSTM OOF not found — stacking without LSTM.")

    results = {}

    for horizon in horizons:
        h_key = f"T{horizon}"
        logger.info(f"\n{'='*40} T+{horizon} — Stacking Ensemble {'='*40}")

        X_meta, y_true = _align_oof_predictions(oof_xgb_lgbm, oof_lstm, h_key)
        if X_meta is None:
            logger.error(f"Cannot build ensemble for {h_key}")
            continue

        n_features = X_meta.shape[1]
        feat_names = ["xgb_proba", "lgbm_proba"]
        if n_features > 2:
            feat_names.append("lstm_proba")

        logger.info(f"  Ensemble input: {len(X_meta)} samples, {n_features} meta-features")
        logger.info(f"  Class distribution: {pd.Series(y_true).value_counts().to_dict()}")

        # Holdout 20% cuối
        val_size = max(1, int(len(X_meta) * 0.2))
        X_train, X_val = X_meta[:-val_size], X_meta[-val_size:]
        y_train, y_val = y_true[:-val_size], y_true[-val_size:]

        # Meta-learner: LogReg với class_weight="balanced" untuk konsistensi
        # dengan XGB/LGBM yang juga handle class imbalance.
        meta = LogisticRegression(
            max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=42
        )
        meta.fit(X_train, y_train)

        # Evaluate on holdout
        y_proba = meta.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1_weighted": float(f1_score(y_val, y_pred, average="weighted", zero_division=0)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        }
        try:
            metrics["auc"] = float(roc_auc_score(y_val, y_proba))
        except ValueError:
            metrics["auc"] = 0.0

        results[h_key] = {"ensemble": {"average": metrics, "n_meta_features": n_features,
                                        "meta_feature_names": feat_names}}

        logger.info(f"  Ensemble: AUC={metrics.get('auc',0):.4f} F1={metrics.get('f1_weighted',0):.4f}")

        # Save meta-learner
        joblib.dump(meta, models_dir / f"ensemble_T{horizon}.pkl")
        logger.info(f"  Saved → {models_dir / f'ensemble_T{horizon}.pkl'}")

    # Save results
    with open(metrics_dir / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Ensemble results saved → {metrics_dir}")
    return results
