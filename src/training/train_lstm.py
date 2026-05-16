"""
VN30 — Bước 4b: LSTM Training (Walk-Forward)

Per CLAUDE.md §5.4:
  - Sequence length: 20 ngày
  - Cùng Walk-Forward scheme với XGB/LGBM
  - Dynamic Thresholding per-fold (same as XGB/LGBM)
  - Purging Gap from config
  - Export OOF predictions (probability)
  - Model class merged từ src/models/lstm_model.py
"""

import os, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from loguru import logger
from tqdm import tqdm

from src.data_collection import filename_to_ticker
from src.label_engineering import (
    apply_purge_embargo,
    compute_dynamic_threshold,
)
from src.training._training_utils import select_multiplier
from src.training.train_xgb_lgbm import create_walk_forward_splits


# ---------------------------------------------------------------------------
# LSTM Network (merged từ src/models/lstm_model.py)
# ---------------------------------------------------------------------------

class _LSTMNetwork(nn.Module):
    """2-layer LSTM architecture."""
    def __init__(self, input_size, units1=64, units2=32, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=units1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=units1, hidden_size=units2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(units2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x); out = self.drop1(out)
        out, _ = self.lstm2(out); out = self.drop2(out)
        return torch.sigmoid(self.fc(out[:, -1, :])).squeeze(-1)


class LSTMModel:
    """LSTM wrapper — train, predict, predict_proba."""
    def __init__(self, units1=64, units2=32, dropout=0.2,
                 epochs=50, batch_size=32, patience=10):
        self.name = "lstm"
        self.units1, self.units2, self.dropout = units1, units2, dropout
        self.epochs, self.batch_size, self.patience = epochs, batch_size, patience
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        if X.ndim != 3: raise ValueError(f"Expected 3D input, got {X.shape}")
        self.model = _LSTMNetwork(X.shape[2], self.units1, self.units2, self.dropout).to(self.device)
        # Early stopping validation: last 10% of train fold (chronological, not random).
        # This is a subset of the current train fold — no data leak across folds.
        val_n = max(1, int(len(X) * 0.1))
        Xt, Xv = torch.tensor(X[:-val_n], dtype=torch.float32, device=self.device), \
                  torch.tensor(X[-val_n:], dtype=torch.float32, device=self.device)
        yt, yv = torch.tensor(y[:-val_n], dtype=torch.float32, device=self.device), \
                  torch.tensor(y[-val_n:], dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters())
        crit = nn.BCELoss()
        best_vl, best_st, pc = float('inf'), None, 0
        for _ in range(self.epochs):
            self.model.train()
            for bx, by in loader:
                opt.zero_grad(); crit(self.model(bx), by).backward(); opt.step()
            self.model.eval()
            with torch.no_grad(): vl = crit(self.model(Xv), yv).item()
            if vl < best_vl: best_vl = vl; best_st = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}; pc = 0
            else:
                pc += 1
                if pc >= self.patience: break
        if best_st: self.model.load_state_dict({k: v.to(self.device) for k, v in best_st.items()})
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32, device=self.device)).cpu().numpy()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _load_features_labels(config):
    pd_dir = config["paths"]["processed_data"]
    feat, lab = {}, {}
    for f in Path(os.path.join(pd_dir, "features")).glob("*_features.parquet"):
        df = pd.read_parquet(f); feat[filename_to_ticker(f.stem)] = df
    for f in Path(os.path.join(pd_dir, "labels")).glob("*_labels.parquet"):
        df = pd.read_parquet(f); lab[filename_to_ticker(f.stem)] = df
    return feat, lab


def _prepare_sequences(all_features, all_labels, horizon=1, seq_len=20):
    """Build 3D sequences from features + continuous returns."""
    rc = f"future_return_t{horizon}"
    all_X, all_ret, all_close, all_d, all_t = [], [], [], [], []

    for t in all_features:
        if t not in all_labels:
            continue
        feat = all_features[t]
        lab = all_labels[t]
        if rc not in lab.columns or "close" not in lab.columns:
            continue
        ci = feat.index.intersection(lab.index)
        feat, lab_sub = feat.loc[ci], lab.loc[ci]
        feat = feat.ffill(limit=5).fillna(0)
        fv = feat.values.astype(np.float32)

        for i in range(seq_len - 1, len(fv)):
            ret_val = lab_sub[rc].iloc[i]
            if pd.isna(ret_val):
                continue
            all_X.append(fv[i - seq_len + 1: i + 1])
            all_ret.append(float(ret_val))
            all_close.append(float(lab_sub["close"].iloc[i]))
            all_d.append(ci[i])
            all_t.append(t)

    return (np.array(all_X, dtype=np.float32), np.array(all_ret, dtype=np.float32),
            np.array(all_close, dtype=np.float64), all_d, all_t)


def _scale_3d(X_tr, X_te):
    n_f = X_tr.shape[2]; sc = StandardScaler()
    Xts = sc.fit_transform(X_tr.reshape(-1, n_f)).reshape(X_tr.shape)
    Xtes = sc.transform(X_te.reshape(-1, n_f)).reshape(X_te.shape)
    return Xts.astype(np.float32), Xtes.astype(np.float32), sc


def compute_metrics(y_true, y_pred, y_proba):
    m = {"accuracy": float(accuracy_score(y_true, y_pred)),
         "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
         "precision": float(precision_score(y_true, y_pred, zero_division=0)),
         "recall": float(recall_score(y_true, y_pred, zero_division=0))}
    try: m["auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError: m["auc"] = 0.0
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_lstm_training(config):
    """Train LSTM for all horizons with Walk-Forward + Dynamic Thresholding."""
    horizons = config["labels"]["horizons"]
    label_cfg = config["labels"]
    seq_len = config["training"].get("sequence_length", 20)
    purging_gap = config["training"].get("purging_gap", 10)
    lstm_cfg = config.get("models", {}).get("lstm", {})
    all_features, all_labels = _load_features_labels(config)
    results, all_oof = {}, {}

    for horizon in horizons:
        logger.info(f"\n{'='*40} T+{horizon} — LSTM {'='*40}")
        X, returns_arr, close_arr, dates, tickers = _prepare_sequences(
            all_features, all_labels, horizon, seq_len)
        dates_arr = pd.DatetimeIndex(dates)
        tickers_arr = np.array(tickers)

        folds = create_walk_forward_splits(sorted(dates_arr.unique()), config)
        if not folds: continue

        fm, oof_p = [], []
        mdl = LSTMModel(
            units1=lstm_cfg.get("units_layer1", 64),
            units2=lstm_cfg.get("units_layer2", 32),
            dropout=lstm_cfg.get("dropout", 0.2),
            epochs=lstm_cfg.get("epochs", 50),
            batch_size=lstm_cfg.get("batch_size", 32),
            patience=lstm_cfg.get("early_stopping_patience", 10),
        )

        window = label_cfg.get("rolling_window", 20)
        init_mult = label_cfg.get("init_multiplier", 0.5)
        max_nan = label_cfg.get("max_nan_ratio", 0.5)
        min_mult = label_cfg.get("min_multiplier", 0.1)
        step_mult = label_cfg.get("multiplier_step", 0.05)

        for fold in tqdm(folds, desc="  LSTM"):
            tr_m = (dates_arr >= fold["train_start"]) & (dates_arr <= fold["train_end"])
            te_m = (dates_arr >= fold["test_start"]) & (dates_arr <= fold["test_end"])
            if tr_m.sum() == 0 or te_m.sum() == 0:
                continue

            # Dynamic thresholding per-ticker on train data
            y_train = np.full(tr_m.sum(), np.nan)
            y_test = np.full(te_m.sum(), np.nan)
            tr_indices = np.where(tr_m)[0]
            te_indices = np.where(te_m)[0]

            for ticker_val in np.unique(tickers_arr[tr_m]):
                # Train
                t_tr = np.array([i for i, idx in enumerate(tr_indices)
                                 if tickers_arr[idx] == ticker_val])
                # Dùng DatetimeIndex thực từ dates_arr để rolling trong
                # compute_dynamic_threshold hoạt động đúng theo thứ tự thời gian.
                close_t = pd.Series(
                    close_arr[tr_indices[t_tr]],
                    index=dates_arr[tr_indices[t_tr]],
                )
                ret_t = returns_arr[tr_indices[t_tr]]
                mult = select_multiplier(close_t, window, init_mult, horizon, max_nan, min_mult, step_mult)
                thresh = compute_dynamic_threshold(close_t, window, mult)
                labs = np.full(len(t_tr), np.nan)
                if len(thresh) == len(ret_t):
                    labs[ret_t > thresh.values] = 1
                    labs[ret_t < -thresh.values] = 0
                else:
                    logger.warning(f"thresh/ret length mismatch for {ticker_val} train — skipping ticker in fold")
                y_train[t_tr] = labs

                # Test (same multiplier)
                t_te = np.array([i for i, idx in enumerate(te_indices)
                                 if tickers_arr[idx] == ticker_val])
                if len(t_te) > 0:
                    close_te = pd.Series(
                        close_arr[te_indices[t_te]],
                        index=dates_arr[te_indices[t_te]],
                    )
                    ret_te = returns_arr[te_indices[t_te]]
                    thresh_te = compute_dynamic_threshold(close_te, window, mult)
                    labs_te = np.full(len(t_te), np.nan)
                    if len(thresh_te) == len(ret_te):
                        labs_te[ret_te > thresh_te.values] = 1
                        labs_te[ret_te < -thresh_te.values] = 0
                    else:
                        logger.warning(f"thresh/ret length mismatch for {ticker_val} test — skipping ticker in fold")
                    y_test[t_te] = labs_te

            # Apply purge/embargo on train labels.
            # Catution: dates_arr[tr_m] có duplicate dates (multi-ticker).
            # apply_purge_embargo nhận Series với DatetimeIndex — implementation
            # phải handle duplicate index bằng cách mask theo date range, không theo positional index.
            y_train_s = pd.Series(y_train, index=dates_arr[tr_m])
            y_train_s = apply_purge_embargo(
                y_train_s, [(fold["train_end"], fold["test_start"])],
                horizon, embargo_days=purging_gap)

            # valid_tr phải được build từ y_train_s SAU purge để length khớp với X[tr_m].
            # y_train_s sau apply_purge_embargo vẫn giữ nguyên length (NaN fill, không drop rows).
            # Nếu implementation của apply_purge_embargo drop rows thay vì NaN-fill,
            # cần align lại theo index. Ở đây ta align an toàn qua reindex.
            y_train_aligned = y_train_s.reindex(pd.DatetimeIndex(dates_arr[tr_m]))
            valid_tr = y_train_aligned.notna().values   # numpy bool, length == tr_m.sum()
            valid_te = ~np.isnan(y_test)                # numpy bool, length == te_m.sum()

            Xtr = X[tr_m][valid_tr]
            ytr = y_train_aligned.values[valid_tr].astype(int)
            Xte = X[te_m][valid_te]
            yte = y_test[valid_te].astype(int)

            if len(Xtr) == 0 or len(Xte) == 0:
                continue

            Xts, Xtes, sc = _scale_3d(Xtr, Xte)

            # Save scaler for inference
            models_dir = Path(config["paths"]["models_output"])
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(sc, models_dir / f"scaler_lstm_T{horizon}_fold{fold['fold_id']}.pkl")

            mdl.fit(Xts, ytr)
            
            # Save model per fold to match XGB/LGBM
            if mdl.model:
                torch.save(mdl.model.state_dict(), models_dir / f"lstm_T{horizon}_fold{fold['fold_id']}.pt")

            yp = mdl.predict_proba(Xtes)
            ypd = (yp >= 0.5).astype(int)
            met = compute_metrics(yte, ypd, yp)
            met["fold_id"] = fold["fold_id"]
            fm.append(met)
            oof_p.append({
                "dates": dates_arr[te_m][valid_te].tolist(),
                "tickers": tickers_arr[te_m][valid_te].tolist(),
                "y_true": yte.tolist(), "y_proba": yp.tolist()
            })

        avg = {}
        if fm:
            for k in ["accuracy", "f1_weighted", "precision", "recall", "auc"]:
                if k in fm[0]:
                    avg[k] = float(np.mean([f[k] for f in fm]))
        results[f"T{horizon}"] = {"lstm": {"average": avg, "per_fold": fm}}
        all_oof[f"T{horizon}"] = {"lstm": oof_p}
        logger.info(f"  LSTM: AUC={avg.get('auc',0):.4f} F1={avg.get('f1_weighted',0):.4f}")

    # Save results
    md = Path(config["paths"]["metrics_output"]); md.mkdir(parents=True, exist_ok=True)
    def _c(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, pd.Timestamp): return str(o)
        return o
    with open(md / "lstm_results.json", "w") as f: json.dump(results, f, indent=2, default=_c)
    with open(md / "lstm_oof.pkl", "wb") as f: pickle.dump(all_oof, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"LSTM results saved → {md}")
    return {"results": results, "oof": all_oof}
