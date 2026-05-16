"""
VN30 — Bước 5: Export JSON Contract cho LLM Agent (v1.2)

Theo CLAUDE.md §5: Sau mỗi lần train, export JSON chuẩn hóa.
Schema bắt buộc — LLM Agent phụ thuộc vào nó.

Output: outputs/predictions/{ticker_filename}_latest.json
  VD: FPT_VN_latest.json  (ticker_to_filename("FPT.VN") → "FPT_VN")

"""

import json
import os
import pickle
from datetime import datetime, date, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.data_collection import filename_to_ticker, ticker_to_filename

# Số ngày calendar tối đa để coi prediction là "fresh" (không stale).
# 4 ngày bao phủ gap cuối tuần (thứ Sáu → thứ Hai = 3 ngày) + 1 ngày lễ.
_STALE_THRESHOLD_DAYS = 4

# Key "ensemble" bị exclude khỏi per_model và model_agreement (CLAUDE.md §5 Rule 2)
_META_LEARNER_KEYS = {"ensemble"}


# ---------------------------------------------------------------------------
# Signal helper
# ---------------------------------------------------------------------------

def _get_signal(direction: str, probability: float) -> str:
    """
    BUY khi UP & prob >= 0.60.
    SELL khi DOWN & prob >= 0.60.
    NEUTRAL trong mọi trường hợp còn lại — bao gồm direction không hợp lệ.
    (CLAUDE.md §5 Rule 4)
    """
    if direction == "UP" and probability >= 0.60:
        return "BUY"
    if direction == "DOWN" and probability >= 0.60:
        return "SELL"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# model_agreement helper
# Chỉ đếm base models — exclude _META_LEARNER_KEYS
# ---------------------------------------------------------------------------

def _count_agreement(per_model: dict, final_direction: str) -> str:
    """
    Đếm số base model vote cùng direction với kết quả cuối.
    Returns 'X/Y' string (CLAUDE.md §5 Rule 2).

    Y = tổng số base models thực tế đã chạy (không đếm ensemble/meta-learner).
    X = số base models dự đoán cùng final_direction.

    per_model đầu vào đã được lọc loại ensemble trước khi truyền vào
    (xem build per_model trong export_predictions).
    """
    total = 0
    agree = 0
    for name, pred in per_model.items():
        # Double-check: bỏ qua meta-learner phòng trường hợp caller không lọc
        if name in _META_LEARNER_KEYS:
            continue
        total += 1
        if pred.get("direction") == final_direction:
            agree += 1
    return f"{agree}/{total}" if total > 0 else "0/0"


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_all_results(metrics_dir: Path) -> dict:
    """
    Load và merge tất cả results files.

    Key trong results file phải là "T+1", "T+3" (với dấu +).
    Nếu training pipeline xuất key dạng "T1", normalize về "T+{n}" khi load.

    Log rõ model nào loaded/skipped để debug dễ hơn.
    """
    # Hardcode tên file đã biết. TFT sẽ được thêm khi implement (CLAUDE.md §4 Step 4).
    result_files = [
        "xgb_lgbm_results.json",
        "lstm_results.json",
        "ensemble_results.json",
        "tft_results.json",       # optional — chỉ load nếu file tồn tại
    ]

    merged = {}
    loaded_files = []
    skipped_files = []

    for fname in result_files:
        path = metrics_dir / fname
        if not path.exists():
            skipped_files.append(fname)
            continue

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"  _load_all_results: cannot parse '{fname}': {e}")
            skipped_files.append(fname)
            continue

        for raw_h_key, models in data.items():
            # Normalize h_key về "T+{n}" format
            h_key = _normalize_h_key(raw_h_key)
            if h_key not in merged:
                merged[h_key] = {}
            merged[h_key].update(models)

        loaded_files.append(fname)

    logger.info(
        f"  Results loaded: {loaded_files} | "
        f"Skipped (not found): {skipped_files}"
    )
    return merged


def _load_all_oof(metrics_dir: Path) -> dict:
    """
    Load và merge tất cả OOF prediction files.

    Normalize h_key về "T+{n}" khi load.
    Log rõ file nào loaded/skipped.
    """
    oof_files = [
        "xgb_lgbm_oof.pkl",
        "lstm_oof.pkl",
        "tft_oof.pkl",   # optional
    ]

    merged = {}
    loaded_files = []
    skipped_files = []

    for fname in oof_files:
        path = metrics_dir / fname
        if not path.exists():
            skipped_files.append(fname)
            continue

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning(f"  _load_all_oof: cannot load '{fname}': {e}")
            skipped_files.append(fname)
            continue

        for raw_h_key, models in data.items():
            # Normalize h_key về "T+{n}" format
            h_key = _normalize_h_key(raw_h_key)
            if h_key not in merged:
                merged[h_key] = {}
            merged[h_key].update(models)

        loaded_files.append(fname)

    logger.info(
        f"  OOF loaded:     {loaded_files} | "
        f"Skipped (not found): {skipped_files}"
    )
    return merged


def _normalize_h_key(raw_key: str) -> str:
    """
    Chuẩn hoá horizon key về format "T+{n}".

    Training pipeline có thể xuất "T1" hoặc "T+1" tuỳ version.
    Hàm này đảm bảo toàn bộ internal dict dùng "T+1" / "T+3".

    Ví dụ:
        "T1"  → "T+1"
        "T+1" → "T+1"  (giữ nguyên)
        "T3"  → "T+3"
        "T+3" → "T+3"  (giữ nguyên)
    """
    if raw_key.startswith("T+"):
        return raw_key          # đã đúng format
    if raw_key.startswith("T") and raw_key[1:].isdigit():
        return f"T+{raw_key[1:]}"   # "T1" → "T+1"
    # Không nhận ra format — giữ nguyên và log warning
    logger.warning(
        f"  _normalize_h_key: unrecognized horizon key '{raw_key}'. "
        f"Keeping as-is. Expected formats: 'T1' or 'T+1'."
    )
    return raw_key


# ---------------------------------------------------------------------------
# OOF prediction lookup
# Dùng h_key đã normalize "T+{horizon}"
# ---------------------------------------------------------------------------

def _latest_pred_for_ticker(
    ticker: str,
    horizon: int,
    all_oof: dict,
    model_name: str,
    direction_threshold: float = 0.5,
) -> dict | None:
    """
    Lấy dự đoán mới nhất của ticker từ fold cuối của model.

    h_key = f"T+{horizon}" (có dấu +) để khớp với
    key đã normalize trong all_oof.

    direction_threshold: ngưỡng phân loại UP/DOWN. Mặc định 0.5; nên
    truyền từ config để nhất quán với training pipeline.

    Returns dict với keys: direction, probability, signal
    hoặc None nếu không có dữ liệu.
    """
    # Dùng "T+{horizon}" — nhất quán với _normalize_h_key
    h_key = f"T+{horizon}"
    preds = all_oof.get(h_key, {}).get(model_name, [])
    if not preds:
        return None

    last_fold = preds[-1]
    if not isinstance(last_fold, dict):
        logger.warning(
            f"  _latest_pred_for_ticker: unexpected fold format for "
            f"model='{model_name}' h_key='{h_key}' — expected dict, got {type(last_fold).__name__}."
        )
        return None

    fold_tickers = last_fold.get("tickers")
    fold_proba   = last_fold.get("y_proba")
    if not fold_tickers or fold_proba is None:
        logger.warning(
            f"  _latest_pred_for_ticker: missing 'tickers' or 'y_proba' key "
            f"for model='{model_name}' h_key='{h_key}'."
        )
        return None

    indices = [i for i, t in enumerate(fold_tickers) if t == ticker]
    if not indices:
        return None

    idx = indices[-1]
    try:
        proba = float(fold_proba[idx])
    except (IndexError, TypeError, ValueError) as e:
        logger.warning(
            f"  _latest_pred_for_ticker: cannot read y_proba[{idx}] "
            f"for model='{model_name}' ticker='{ticker}': {e}"
        )
        return None
    direction = "UP" if proba >= direction_threshold else "DOWN"
    return {
        "direction": direction,
        "probability": round(proba, 4),
        "signal": _get_signal(direction, proba),
    }


# ---------------------------------------------------------------------------
# is_stale helper
# Tính động từ data_date thay vì hardcode False
# ---------------------------------------------------------------------------

def _compute_is_stale(data_date_str: str, threshold_days: int = _STALE_THRESHOLD_DAYS) -> bool:
    """
    Trả về True nếu data_date cách hôm nay quá threshold_days ngày.

    Thay thế hardcode `is_stale: False`.
    Dùng calendar days (không phải business days) để đơn giản;
    threshold=4 bao phủ gap cuối tuần + 1 ngày lễ.
    """
    if not data_date_str:
        return True     # không có data_date → coi là stale
    try:
        data_dt = date.fromisoformat(data_date_str)
        delta = (date.today() - data_dt).days
        return delta > threshold_days
    except ValueError:
        logger.warning(f"  _compute_is_stale: invalid data_date '{data_date_str}' → marking stale")
        return True


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_predictions(config: dict) -> None:
    """
    Export {{ticker_filename}}_latest.json cho tất cả tickers.

    JSON schema theo CLAUDE.md §5 (bắt buộc, không thay đổi).
    Tên file: ticker_to_filename(ticker) + "_latest.json"
      VD: "FPT.VN" → "FPT_VN_latest.json"

    Dùng ticker_to_filename() để tránh dấu "." trong tên file.
    """
    metrics_dir = Path(config["paths"]["metrics_output"])
    pred_dir = Path(config["paths"].get("predictions_output", "outputs/predictions"))
    pred_dir.mkdir(parents=True, exist_ok=True)

    # --- Load training artifacts ---
    logger.info("Loading training results and OOF predictions...")
    all_results = _load_all_results(metrics_dir)
    all_oof = _load_all_oof(metrics_dir)

    if not all_results:
        logger.error("No training results found. Run training first.")
        return

    horizons: list[int] = config["labels"]["horizons"]
    direction_threshold: float = float(
        config.get("export", {}).get("direction_threshold", 0.5)
    )

    # --- Load feature files ---
    processed_dir = config["paths"]["processed_data"]
    feat_dir = Path(processed_dir) / "features"
    all_features: dict[str, pd.DataFrame] = {}

    for feat_file in feat_dir.glob("*_features.parquet"):
        try:
            df = pd.read_parquet(feat_file)
        except Exception as e:
            logger.warning(f"Cannot read feature file '{feat_file.name}': {e}")
            continue

        # Primary: parse ticker từ tên file (pattern: "{TICKER_VN}_features.parquet")
        # filename_to_ticker v3.1 raise ValueError nếu không nhận ra format.
        try:
            ticker = filename_to_ticker(feat_file.stem.replace("_features", ""))
        except ValueError as e:
            logger.error(f"  Cannot determine ticker from filename '{feat_file.name}': {e} — skipping.")
            continue

        # Validation: nếu cột "ticker" có mặt, kiểm tra nhất quán với tên file.
        # Cột "ticker" thuộc OHLCV schema — có thể không có trong feature parquet.
        if "ticker" in df.columns and not df["ticker"].empty:
            col_ticker = str(df["ticker"].iloc[0])
            if col_ticker and col_ticker != ticker:
                logger.warning(
                    f"  '{feat_file.name}': filename ticker='{ticker}' != "
                    f"column ticker='{col_ticker}'. Using filename value."
                )

        all_features[ticker] = df

    if not all_features:
        logger.error("No feature files found. Run feature engineering first.")
        return

    logger.info(f"Loaded features for {len(all_features)} tickers.")

    # --- Xác định best BASE model per horizon (theo AUC) ---
    # Exclude "ensemble" khỏi việc chọn best model.
    # "ensemble" là meta-learner, không so sánh công bằng với base models.
    best_per_h: dict[int, str] = {}
    for h in horizons:
        h_key = f"T+{h}"
        hr = all_results.get(h_key, {})
        # Chỉ xét base models (không phải meta-learner)
        base_models = {n: v for n, v in hr.items() if n not in _META_LEARNER_KEYS}
        if base_models:
            best_name = max(
                base_models,
                key=lambda n: base_models[n].get("average", {}).get("auc", 0),
            )
            best_per_h[h] = best_name
            logger.info(f"  Best base model for T+{h}: '{best_name}'")
        else:
            logger.warning(f"  No base models found for T+{h} — skipping horizon.")

    # --- Export per ticker ---
    all_tickers = sorted(all_features.keys())
    exported = 0
    skipped = 0

    for ticker in all_tickers:
        feat_df = all_features[ticker]

        # Data date (ngày cuối cùng trong feature index)
        data_date = ""
        if not feat_df.empty:
            data_date = str(feat_df.index.max().date())

        # Feature values mới nhất (dùng để điền "value" trong top_features)
        feat_vals: dict = {}
        if not feat_df.empty:
            feat_vals = feat_df.iloc[-1].to_dict()

        # --- Build predictions per horizon ---
        predictions: dict = {}

        for h in horizons:
            h_key = f"T+{h}"
            best_name = best_per_h.get(h)
            if not best_name:
                continue

            hr = all_results.get(h_key, {})

            # Metrics của best base model cho horizon này
            # None (→ JSON null) khi không có dữ liệu, để LLM Agent phân biệt
            # "accuracy thực sự = 0" vs "chưa có metrics".
            _acc_raw = hr.get(best_name, {}).get("average", {}).get("accuracy")
            acc = round(float(_acc_raw), 4) if _acc_raw is not None else None
            sharpe = hr.get(best_name, {}).get("average", {}).get("sharpe", None)

            # per_model chỉ gồm BASE models — exclude ensemble/meta-learner.
            # Iterate qua hr (models có trong results) nhưng lọc _META_LEARNER_KEYS.
            per_model: dict = {}
            for mn in hr:
                if mn in _META_LEARNER_KEYS:
                    continue    # bỏ qua meta-learner
                pred = _latest_pred_for_ticker(ticker, h, all_oof, mn, direction_threshold)
                if pred:
                    per_model[mn] = {
                        "direction":   pred["direction"],
                        "probability": pred["probability"],
                    }

            # Main prediction từ best base model
            main_pred = _latest_pred_for_ticker(ticker, h, all_oof, best_name, direction_threshold)
            if not main_pred:
                logger.debug(
                    f"  [{ticker}] No OOF prediction for best model '{best_name}' "
                    f"at T+{h} — skipping horizon."
                )
                continue

            # model_agreement: chỉ đếm base models (per_model đã lọc ensemble)
            agreement = _count_agreement(per_model, main_pred["direction"])

            # Top features per-horizon — CLAUDE.md §5 Rule 3 (riêng biệt cho T+1 và T+3)
            shap_imp = hr.get(best_name, {}).get("shap_top10", {})
            feat_imp = hr.get(best_name, {}).get("feature_importance", {})
            src = shap_imp if shap_imp else feat_imp
            top_features_h: list = []
            if src:
                top5 = sorted(src.items(), key=lambda x: x[1], reverse=True)[:5]
                top_features_h = [
                    {
                        "name":       n,
                        "value":      round(float(feat_vals.get(n, 0.0)), 4),
                        "importance": round(float(v), 4),
                    }
                    for n, v in top5
                ]

            # backtest_sharpe: fix falsy check cho sharpe=0.0 và np.nan
            # "if sharpe" sai vì: bool(0.0)=False, bool(np.nan)=True
            # None → JSON null: phân biệt "không có backtest" vs "sharpe thực sự = 0.0"
            if sharpe is not None and not np.isnan(float(sharpe)):
                backtest_sharpe = round(float(sharpe), 4)
            else:
                backtest_sharpe = None

            predictions[h_key] = {
                "direction":       main_pred["direction"],
                "probability":     main_pred["probability"],
                "signal":          main_pred["signal"],
                "model_agreement": agreement,
                "model_accuracy":  acc,
                "backtest_sharpe": backtest_sharpe,
                "top_features":    top_features_h,
                "per_model":       per_model,
            }

        if not predictions:
            logger.debug(f"  [{ticker}] No predictions built — skipping export.")
            skipped += 1
            continue

        # --- Build payload ---
        now = datetime.now(timezone.utc)

        # is_stale tính động từ data_date
        is_stale = _compute_is_stale(data_date)
        if is_stale:
            logger.warning(f"  [{ticker}] data_date='{data_date}' is stale (>{_STALE_THRESHOLD_DAYS}d old).")

        payload = {
            "ticker":       ticker,           # "FPT.VN" — chuẩn CLAUDE.md §4
            "generated_at": now.isoformat(),
            "data_date":    data_date,
            "is_stale":     is_stale,
            "predictions":  predictions,
        }

        # Tên file: ticker_to_filename("FPT.VN") → "FPT_VN"
        # Tránh dấu "." trong tên file trước "_latest.json"
        safe_name = ticker_to_filename(ticker)
        out_path = pred_dir / f"{safe_name}_latest.json"

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            exported += 1
            logger.debug(f"  [{ticker}] → {out_path.name}")
        except Exception as e:
            logger.error(f"  [{ticker}] Failed to write JSON: {e}")
            skipped += 1

    logger.info(
        f"\nExport complete: {exported} exported, {skipped} skipped "
        f"→ {pred_dir}"
    )