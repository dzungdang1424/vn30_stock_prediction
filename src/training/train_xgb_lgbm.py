"""
VN30 — Bước 4a: XGBoost + LightGBM Training (Walk-Forward)

Per CLAUDE.md §5.4:
  - Walk-Forward Validation (config.yaml)
  - Dynamic Thresholding: select_multiplier() runs INSIDE each fold
  - Purging Gap from config (default 10 days)
  - Export OOF predictions cho stacking
  - SHAP values top-10
  - Checkpoint sau mỗi ticker
"""

import os, json, pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from tqdm import tqdm

try:
    import shap; HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.data_collection import filename_to_ticker
from src.label_engineering import (
    apply_purge_embargo,
    compute_dynamic_threshold,
)
from src.training._training_utils import select_multiplier


def _create_xgboost(config):
    p = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
         "random_state": 42, "eval_metric": "logloss", "verbosity": 0}
    p.update(config.get("models", {}).get("xgboost", {}))
    return XGBClassifier(**p)


def _create_lightgbm(config):
    p = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
         "random_state": 42, "class_weight": "balanced", "verbose": -1, "n_jobs": -1}
    p.update(config.get("models", {}).get("lightgbm", {}))
    return LGBMClassifier(**p)


def create_walk_forward_splits(dates, config):
    t = config["training"]
    td, ted, sd = t.get("train_days", 252), t.get("test_days", 63), t.get("step_days", 63)
    mode = t.get("walk_forward_mode", "rolling")
    dates = sorted(dates); total = len(dates); folds = []; start = 0
    while start + td + ted <= total:
        ts = dates[0] if mode == "expanding" else dates[start]
        te = dates[start + td - 1]
        tes = dates[start + td]
        tee = dates[min(start + td + ted - 1, total - 1)]
        folds.append({"fold_id": len(folds), "train_start": ts, "train_end": te,
                       "test_start": tes, "test_end": tee})
        start += sd
    logger.info(f"Created {len(folds)} walk-forward folds (mode={mode})")
    return folds


def compute_metrics(y_true, y_pred, y_proba):
    m = {"accuracy": float(accuracy_score(y_true, y_pred)),
         "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
         "precision": float(precision_score(y_true, y_pred, zero_division=0)),
         "recall": float(recall_score(y_true, y_pred, zero_division=0))}
    try: m["auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError: m["auc"] = 0.0
    return m


def _simple_backtest_sharpe(y_true, y_proba):
    signals = (y_proba >= 0.5).astype(int)
    returns = np.where(signals == 1, np.where(y_true == 1, 0.01, -0.01), 0.0)
    if returns.std() == 0: return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))


def _load_features_labels(config):
    pd_dir = config["paths"]["processed_data"]
    feat, lab = {}, {}
    for f in Path(os.path.join(pd_dir, "features")).glob("*_features.parquet"):
        df = pd.read_parquet(f)
        feat[filename_to_ticker(f.stem)] = df
    for f in Path(os.path.join(pd_dir, "labels")).glob("*_labels.parquet"):
        df = pd.read_parquet(f)
        lab[filename_to_ticker(f.stem)] = df
    if not feat or not lab:
        raise RuntimeError("Features or labels not found.")
    return feat, lab


def _prepare_ml_data(all_features, all_labels, horizon=1):
    """Join features with continuous returns + close prices."""
    rc = f"future_return_t{horizon}"
    frames = []
    for t in all_features:
        if t not in all_labels:
            continue
        label_df = all_labels[t]
        cols_needed = [c for c in [rc, "close"] if c in label_df.columns]
        if rc not in cols_needed:
            continue
        m = all_features[t].copy().join(label_df[cols_needed], how="inner")
        m = m.dropna(subset=[rc])
        m["ticker"] = t
        frames.append(m)
    if not frames:
        raise RuntimeError("No valid data after joining features and labels.")
    c = pd.concat(frames).sort_index()
    fc = [col for col in c.columns if col not in [rc, "close", "ticker"]]
    return c[fc], c[rc], c["close"], c["ticker"], c.index


def _binarize_per_ticker(close_series, returns_series, tickers_series,
                         mask, label_cfg, horizon):
    """
    Dynamic thresholding per-ticker within a date mask.

    For each ticker: select_multiplier on its close subset,
    then compute binary labels from returns + threshold.
    Returns a Series of binary labels (1/0/NaN).
    """
    idx = close_series[mask].index
    labels = pd.Series(np.nan, index=idx)
    multipliers = {}

    window = label_cfg.get("rolling_window", 20)
    init_mult = label_cfg.get("init_multiplier", 0.5)
    max_nan = label_cfg.get("max_nan_ratio", 0.5)
    min_mult = label_cfg.get("min_multiplier", 0.1)
    step_mult = label_cfg.get("multiplier_step", 0.05)

    for ticker_val in tickers_series[mask].unique():
        t_mask = mask & (tickers_series == ticker_val)
        close_t = close_series[t_mask]
        returns_t = returns_series[t_mask]

        mult = select_multiplier(
            close_train=close_t, window=window,
            init_multiplier=init_mult, horizon=horizon,
            max_nan_ratio=max_nan, min_multiplier=min_mult,
            multiplier_step=step_mult,
        )
        multipliers[ticker_val] = mult

        threshold = compute_dynamic_threshold(close_t, window, mult)
        # Binarize trực tiếp từ returns vs threshold — không dùng create_labels()
        # vì create_labels() dùng shift(-horizon) gây lookahead bias (CLAUDE.md §3).
        t_labels = pd.Series(np.nan, index=close_t.index)
        t_labels[returns_t > threshold] = 1
        t_labels[returns_t < -threshold] = 0
        labels.loc[close_t.index] = t_labels

    return labels, multipliers


def _binarize_test_with_multipliers(close_series, returns_series, tickers_series,
                                     mask, multipliers, label_cfg):
    """Apply pre-computed multipliers to test data."""
    idx = close_series[mask].index
    labels = pd.Series(np.nan, index=idx)
    window = label_cfg.get("rolling_window", 20)

    for ticker_val, mult in multipliers.items():
        t_mask = mask & (tickers_series == ticker_val)
        if not t_mask.any():
            continue
        close_t = close_series[t_mask]
        returns_t = returns_series[t_mask]
        threshold = compute_dynamic_threshold(close_t, window, mult)
        t_labels = pd.Series(np.nan, index=close_t.index)
        t_labels[returns_t > threshold] = 1
        t_labels[returns_t < -threshold] = 0
        labels.loc[close_t.index] = t_labels

    return labels


def run_xgb_lgbm_training(config):
    """Main entry — train XGB + LGBM for all horizons with dynamic thresholding."""
    horizons = config["labels"]["horizons"]
    label_cfg = config["labels"]
    purging_gap = config["training"].get("purging_gap", 10)
    all_features, all_labels = _load_features_labels(config)
    results, all_oof = {}, {}

    for horizon in horizons:
        logger.info(f"\n{'='*40} T+{horizon} — XGB+LGBM {'='*40}")
        X, returns_s, close_s, tickers_s, dates_idx = _prepare_ml_data(
            all_features, all_labels, horizon)
        dates_arr = pd.DatetimeIndex(dates_idx)
        folds = create_walk_forward_splits(sorted(dates_arr.unique()), config)
        if not folds:
            logger.error(f"No folds for T+{horizon}"); continue

        h_results, h_oof = {}, {}
        for mn, model_fn in [("xgboost", lambda: _create_xgboost(config)),
                              ("lightgbm", lambda: _create_lightgbm(config))]:
            logger.info(f"\n  Training {mn} (T+{horizon})...")
            fm, oof_p, imps, shap_accum = [], [], [], []

            for fold in tqdm(folds, desc=f"  {mn}"):
                tr_m = (dates_arr >= fold["train_start"]) & (dates_arr <= fold["train_end"])
                te_m = (dates_arr >= fold["test_start"]) & (dates_arr <= fold["test_end"])

                if tr_m.sum() == 0 or te_m.sum() == 0:
                    continue

                # Dynamic thresholding per-ticker on train data (CLAUDE.md §5.4)
                y_train_binary, multipliers = _binarize_per_ticker(
                    close_s, returns_s, tickers_s, tr_m, label_cfg, horizon)

                # Apply purge/embargo on train labels
                y_train_purged = apply_purge_embargo(
                    y_train_binary,
                    [(fold["train_end"], fold["test_start"])],
                    horizon, embargo_days=purging_gap)

                # Binarize test with same multipliers from train
                y_test_binary = _binarize_test_with_multipliers(
                    close_s, returns_s, tickers_s, te_m, multipliers, label_cfg)

                # Filter valid samples — dùng boolean numpy array thay vì .loc
                # để tránh ambiguous indexing khi DatetimeIndex có duplicate dates
                # (multi-ticker cùng ngày).
                valid_tr_mask = y_train_purged.notna().values  # numpy bool array
                valid_te_mask = y_test_binary.notna().values   # numpy bool array

                X_tr_raw = X[tr_m].values[valid_tr_mask]
                yt = y_train_purged.values[valid_tr_mask].astype(int)
                X_te_raw = X[te_m].values[valid_te_mask]
                yts = y_test_binary.values[valid_te_mask].astype(int)

                if len(X_tr_raw) == 0 or len(X_te_raw) == 0:
                    continue

                # Scale: fit on TRAIN, transform both
                sc = StandardScaler()
                X_tr_scaled = sc.fit_transform(X_tr_raw)
                X_te_scaled = sc.transform(X_te_raw)

                # Save scaler for inference
                models_dir = Path(config["paths"]["models_output"])
                models_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(sc, models_dir / f"scaler_{mn}_T{horizon}_fold{fold['fold_id']}.pkl")

                # Train on TRAIN scaled data
                model = model_fn()
                if mn == "xgboost":
                    np_, nn_ = (yt == 1).sum(), (yt == 0).sum()
                    if np_ > 0:
                        model.set_params(scale_pos_weight=nn_ / np_)
                model.fit(X_tr_scaled, yt)
                joblib.dump(model, models_dir / f"{mn}_T{horizon}_fold{fold['fold_id']}.pkl")

                # Predict on TEST scaled data
                yp = model.predict_proba(X_te_scaled)[:, 1]
                ypd = (yp >= 0.5).astype(int)
                met = compute_metrics(yts, ypd, yp)
                met.update({"fold_id": fold["fold_id"],
                            "sharpe": _simple_backtest_sharpe(yts, yp)})
                fm.append(met)
                imps.append(dict(zip(X.columns.tolist(),
                                     model.feature_importances_.tolist())))

                # SHAP: accumulate mean |shap| per fold (over all folds, not just last)
                if HAS_SHAP:
                    try:
                        sv = shap.TreeExplainer(model).shap_values(X_te_scaled)
                        if isinstance(sv, list):
                            sv = sv[1]
                        shap_accum.append(np.abs(sv).mean(axis=0))
                    except Exception as e:
                        logger.warning(f"SHAP failed for fold {fold['fold_id']}: {e}")

                oof_p.append({
                    "dates": dates_arr[te_m][valid_te_mask].tolist(),
                    "tickers": tickers_s[te_m][valid_te_mask].tolist(),
                    "y_true": yts.tolist(), "y_proba": yp.tolist()
                })

            avg = {}
            if fm:
                for k in ["accuracy", "f1_weighted", "precision", "recall", "auc", "sharpe"]:
                    if k in fm[0]:
                        avg[k] = float(np.mean([f[k] for f in fm]))

            # SHAP top-10: mean |shap| averaged across ALL folds, không chỉ fold cuối
            shap_t10 = {}
            if HAS_SHAP and shap_accum:
                mean_shap = np.mean(np.stack(shap_accum, axis=0), axis=0)
                shap_t10 = dict(sorted(
                    zip(X.columns.tolist(), mean_shap.tolist()),
                    key=lambda x: x[1], reverse=True)[:10])

            h_results[mn] = {
                "average": avg, "per_fold": fm, "shap_top10": shap_t10,
                "feature_importance": {
                    k: float(np.mean([i.get(k, 0) for i in imps]))
                    for k in (imps[0] if imps else {})
                }
            }
            h_oof[mn] = oof_p
            logger.info(f"  {mn}: AUC={avg.get('auc',0):.4f} "
                        f"F1={avg.get('f1_weighted',0):.4f} "
                        f"Sharpe={avg.get('sharpe',0):.4f}")

        results[f"T{horizon}"] = h_results
        all_oof[f"T{horizon}"] = h_oof

    # Save
    md = Path(config["paths"]["metrics_output"]); md.mkdir(parents=True, exist_ok=True)
    def _c(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, pd.Timestamp): return str(o)
        return o
    with open(md / "xgb_lgbm_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_c)
    with open(md / "xgb_lgbm_oof.pkl", "wb") as f:
        pickle.dump(all_oof, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Results & OOF saved → {md}")
    return {"results": results, "oof": all_oof}
