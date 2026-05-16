"""
VN30 Stock Prediction Pipeline — Bước 2: Feature Engineering (v3.2)

Tính raw features + cross-sectional rank features từ OHLCV.
Không sử dụng bất kỳ thông tin tương lai nào (no lookahead bias).

Thay đổi v3.2 (TFT support):
  - compute_calendar_features(): 5 calendar features cho TFT TIME_VARYING_KNOWN_REALS
    (dow_sin, dow_cos, month_sin, month_cos, is_quarter_end) — tính từ date index,
    không từ giá. Được merge vào features parquet của từng ticker.
  - compute_static_features(): market_cap_rank cho TFT STATIC_REALS — lưu riêng
    vào data/processed/features/static_features.parquet (không gộp vào time-series).
  - run_feature_engineering(): tích hợp 2 hàm trên, cập nhật dropna logic
    để calendar cols không ảnh hưởng warm-up period filtering.

"""

from pathlib import Path
import pandas as pd
import numpy as np
import pandas_ta as ta
from loguru import logger
from functools import reduce

from src.data_collection import filename_to_ticker, ticker_to_filename


# ---------------------------------------------------------------------------
# Required OHLCV columns và expected dtype (dùng để validate khi load)
# ---------------------------------------------------------------------------
_REQUIRED_OHLCV_COLS: dict[str, str] = {
    "open":   "float64",
    "high":   "float64",
    "low":    "float64",
    "close":  "float64",
    "volume": "int64",
}

# Internal columns (prefix "_"): dùng để tính intermediate/rank, không lưu ra file.
# Tên lowercase nhất quán.
_INTERNAL_COLS: frozenset[str] = frozenset({"_obv", "_pvt", "_atr_raw"})


# ---------------------------------------------------------------------------
# Helper: ép về Series an toàn từ kết quả pandas-ta
# ---------------------------------------------------------------------------
def _to_series(result, fallback_index) -> pd.Series:
    if isinstance(result, pd.DataFrame) and not result.empty:
        return result.iloc[:, 0]
    if isinstance(result, pd.Series):
        return result
    return pd.Series(np.nan, index=fallback_index)


# ---------------------------------------------------------------------------
# Schema validation helper
# ---------------------------------------------------------------------------
def _validate_ohlcv(ticker: str, df: pd.DataFrame) -> list[str]:
    """
    Kiểm tra schema OHLCV trước khi compute features.
    Trả về list lỗi (rỗng = pass).

    Tránh TypeError không được catch bên trong compute_raw_features.
    """
    errors: list[str] = []
    for col, expected_dtype in _REQUIRED_OHLCV_COLS.items():
        if col not in df.columns:
            errors.append(f"Missing required column '{col}'")
        else:
            actual = str(df[col].dtype)
            # Chấp nhận float32/float64 cho price cols; int32/int64 cho volume
            if expected_dtype == "float64" and not pd.api.types.is_float_dtype(df[col]):
                errors.append(f"Column '{col}': dtype={actual}, expected float")
            elif expected_dtype == "int64" and not pd.api.types.is_integer_dtype(df[col]):
                errors.append(f"Column '{col}': dtype={actual}, expected int (will cast)")
    return errors


# ---------------------------------------------------------------------------
# TFT Calendar Features — TIME_VARYING_KNOWN_REALS
# Tính từ date index, hoàn toàn không dùng giá/khối lượng.
# Phân loại TIME_VARYING_KNOWN vì ta biết chính xác các giá trị này
# trong tương lai tại thời điểm predict (chỉ phụ thuộc lịch).
# ---------------------------------------------------------------------------
def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính 5 calendar features cho TFT TIME_VARYING_KNOWN_REALS.

    Input : DataFrame với DatetimeIndex (index = ngày giao dịch).
    Output: DataFrame cùng index, 5 cột mới — KHÔNG thêm cột nào khác.

    Encoding tuần hoàn (sin/cos) thay vì raw integer để tránh discontinuity
    (ví dụ: thứ Sáu=4 và thứ Hai=0 phải gần nhau về mặt khoảng cách).

    Không gọi dropna() — orchestrator quyết định chiến lược.
    """
    idx = df.index
    cal = pd.DataFrame(index=idx)

    # day_of_week: 0=Thứ Hai … 4=Thứ Sáu (thị trường VN không giao dịch T7/CN)
    dow = idx.dayofweek.astype(float)
    cal["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    cal["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # month: 1–12
    month = idx.month.astype(float)
    cal["month_sin"] = np.sin(2 * np.pi * month / 12)
    cal["month_cos"] = np.cos(2 * np.pi * month / 12)

    # is_quarter_end: True vào ngày cuối quý (Q1=31/3, Q2=30/6, Q3=30/9, Q4=31/12)
    # Cast về float để nhất quán dtype với các cột còn lại
    cal["is_quarter_end"] = idx.is_quarter_end.astype(float)

    return cal


# ---------------------------------------------------------------------------
# TFT Static Features — STATIC_REALS
# Tính 1 lần cho toàn bộ lịch sử, lưu riêng vào static_features.parquet.
# Không thay đổi theo ngày — TFT dùng để học entity-level context.
# ---------------------------------------------------------------------------
def compute_static_features(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Tính market_cap_rank cho 30 mã VN30.

    Dùng average daily volume làm proxy cho market cap (tương quan cao
    trên VN30 và không cần dữ liệu bên ngoài OHLCV).

    Input : all_data — dict {ticker: OHLCV DataFrame} của toàn bộ tickers đã load.
    Output: DataFrame với index=ticker, 1 cột "market_cap_rank" ∈ [0, 1].
            rank=1.0 → mã có volume cao nhất (bluechip nhất trong rổ).
            rank=0.0 → mã có volume thấp nhất.

    Lưu vào: data/processed/features/static_features.parquet
    (không gộp vào time-series features parquet để tránh confusion).
    """
    if not all_data:
        raise ValueError("all_data rỗng — không thể tính static features.")

    avg_volumes = {
        ticker: df["volume"].mean()
        for ticker, df in all_data.items()
        if "volume" in df.columns and not df["volume"].isna().all()
    }

    vol_series = pd.Series(avg_volumes, name="avg_volume")

    # pct_rank: normalize về [0, 1], ticker volume cao nhất = 1.0
    static_df = vol_series.rank(pct=True).rename("market_cap_rank").to_frame()
    static_df.index.name = "ticker"

    return static_df


# ---------------------------------------------------------------------------
# Atomic write helper — per-ticker checkpoint
# Thay thế batch-delete-then-save bằng atomic per-ticker
# ---------------------------------------------------------------------------
def _save_feature_checkpoint(
    ticker: str,
    fdf: pd.DataFrame,
    feat_dir: Path,
) -> bool:
    """
    Lưu feature DataFrame ra Parquet với atomic write (tmp → rename).
    Pattern: ghi .tmp → rename live→.bak → rename .tmp→live → xóa .bak.

    Returns True nếu thành công, False nếu lỗi.
    """
    feat_dir.mkdir(parents=True, exist_ok=True)

    filename  = f"{ticker_to_filename(ticker)}_features.parquet"
    live_path = feat_dir / filename
    tmp_path  = feat_dir / f"{filename}.tmp"
    bak_path  = feat_dir / f"{filename}.bak"

    try:
        fdf.to_parquet(tmp_path, compression="zstd")

        if live_path.exists():
            live_path.rename(bak_path)

        tmp_path.rename(live_path)

        if bak_path.exists():
            bak_path.unlink()

        return True

    except Exception as e:
        logger.error(f"  [{ticker}] Failed to save feature checkpoint: {e}")
        if bak_path.exists() and not live_path.exists():
            bak_path.rename(live_path)
            logger.warning(f"  [{ticker}] Rolled back to backup.")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


# ---------------------------------------------------------------------------
# Raw features
# Tất cả tên cột lowercase
# ---------------------------------------------------------------------------
def compute_raw_features(df: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """
    Tính raw features cho một ticker từ OHLCV DataFrame đã validate.

    Không thực hiện scaling (NOTE-FE3): StandardScaler/MinMaxScaler phải
    fit trên train fold rồi transform test fold riêng — thực hiện trong
    data_preparation.py.

    Tất cả tên cột output là lowercase hoàn toàn.

    Returns:
        DataFrame với index = ngày giao dịch, columns = tên feature lowercase.
        Các hàng đầu có NaN do warm-up period là bình thường.
        dropna() KHÔNG được gọi ở đây — orchestrator quyết định chiến lược.
    """
    idx = df.index
    features = pd.DataFrame(index=idx)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    volume = df["volume"].astype("float64")   # cast để rolling/divide hoạt động

    close_safe = close.replace(0, np.nan)

    # ── MOMENTUM ──────────────────────────────────────────────────────────
    # "RVI" → "rvi"
    features["rvi"] = _to_series(
        ta.rvgi(open_=open_, high=high, low=low, close=close), idx
    )

    # kama_ratio = KAMA / close (scale-free, FIX-FE6)
    # "KAMA_ratio" → "kama_ratio"
    kama_raw = _to_series(
        ta.kama(close=close, length=feat_cfg.get("kama_period", 10)), idx
    )
    features["kama_ratio"] = kama_raw / close_safe

    # "RSI" → "rsi"
    features["rsi"] = _to_series(
        ta.rsi(close=close, length=feat_cfg.get("rsi_period", 14)), idx
    )

    # MACD — [CRITICAL-1] "MACD_HIST" → "macd_hist", "MACD_SIG" → "macd_sig"
    macd_df = ta.macd(close=close)
    if isinstance(macd_df, pd.DataFrame) and not macd_df.empty:
        hist_col = next((c for c in macd_df.columns if "MACDh" in str(c)), None)
        sig_col  = next((c for c in macd_df.columns if "MACDs" in str(c)), None)
        features["macd_hist"] = macd_df[hist_col] if hist_col else np.nan
        features["macd_sig"]  = macd_df[sig_col]  if sig_col  else np.nan
    else:
        features["macd_hist"] = np.nan
        features["macd_sig"]  = np.nan

    # ── TREND & SERIES DECOMPOSITION ──────────────────────────────────────
    # tema_ratio, hma_ratio (scale-free, FIX-FE6)
    # "TEMA_ratio" → "tema_ratio", "HMA_ratio" → "hma_ratio"
    tema_raw = _to_series(
        ta.tema(close=close, length=feat_cfg.get("tema_period", 20)), idx
    )
    hma_raw = _to_series(
        ta.hma(close=close, length=feat_cfg.get("hma_period", 20)), idx
    )
    features["tema_ratio"] = tema_raw / close_safe
    features["hma_ratio"]  = hma_raw  / close_safe

    # Series decomposition: Trend ratio + Residual ratio
    # "DECOMP_TREND" → "decomp_trend", "DECOMP_RESIDUAL" → "decomp_residual"
    trend_period = feat_cfg.get("trend_period", 20)
    sma_raw = _to_series(ta.sma(close=close, length=trend_period), idx)
    features["decomp_trend"]    = sma_raw / close_safe
    features["decomp_residual"] = (close - sma_raw) / close_safe

    # Mass Index — [CRITICAL-1] "MI" → "mi"
    features["mi"] = _to_series(ta.massi(high=high, low=low), idx)

    # ── ICHIMOKU (backward-looking only, scale-free) ───────────────────────
    # Tenkan-sen và Kijun-sen chỉ dùng rolling max/min trên dữ liệu quá khứ.
    # Senkou Span A/B và Chikou Span KHÔNG được dùng (forward-looking).
    # "ICH_TENKAN_ratio" → "ich_tenkan_ratio", etc.
    tenkan = feat_cfg.get("ichimoku_tenkan", 9)
    kijun  = feat_cfg.get("ichimoku_kijun", 26)

    tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_line  = (high.rolling(kijun).max()  + low.rolling(kijun).min())  / 2

    features["ich_tenkan_ratio"] = tenkan_line / close_safe
    features["ich_kijun_ratio"]  = kijun_line  / close_safe

    # ATR (scale-free FIX-FE6): atr_pct = ATR/close
    # _atr_raw là internal column, dùng để tính ich_dist_kijun rồi drop.
    # "ATR_pct" → "atr_pct", "_ATR_raw" → "_atr_raw"
    atr_raw = _to_series(
        ta.atr(high=high, low=low, close=close, length=feat_cfg.get("atr_period", 14)), idx
    )
    features["atr_pct"]   = atr_raw / close_safe
    features["_atr_raw"]  = atr_raw                # internal — dropped in orchestrator

    # ich_dist_kijun = (close - kijun) / ATR (scale-free, đã đúng từ v3.0)
    # "ICH_DIST_KIJUN" → "ich_dist_kijun"
    features["ich_dist_kijun"] = (close - kijun_line) / atr_raw.replace(0, np.nan)

    # ── VOLATILITY ────────────────────────────────────────────────────────
    # "BB_PCT" → "bb_pct", "BB_WIDTH" → "bb_width"
    bb = ta.bbands(
        close=close,
        length=feat_cfg.get("bb_period", 20),
        std=feat_cfg.get("bb_std", 2.0),
    )
    if isinstance(bb, pd.DataFrame) and not bb.empty:
        upper_col = next((c for c in bb.columns if "BBU" in str(c).upper()), None)
        lower_col = next((c for c in bb.columns if "BBL" in str(c).upper()), None)
        mid_col   = next((c for c in bb.columns if "BBM" in str(c).upper()), None)

        if upper_col and lower_col:
            bb_upper = bb[upper_col]
            bb_lower = bb[lower_col]
            bb_mid   = bb[mid_col] if mid_col else close.rolling(20).mean()
            bb_range = (bb_upper - bb_lower).replace(0, np.nan)

            features["bb_pct"]   = (close - bb_lower) / bb_range
            features["bb_width"] = bb_range / bb_mid
        else:
            features["bb_pct"] = features["bb_width"] = np.nan
    else:
        features["bb_pct"] = features["bb_width"] = np.nan

    # ── VOLUME ────────────────────────────────────────────────────────────
    # "MFI" → "mfi"
    features["mfi"] = _to_series(
        ta.mfi(high=high, low=low, close=close, volume=volume,
               length=feat_cfg.get("mfi_period", 14)), idx
    )

    # OBV / PVT — normalize về ratio để scale-free
    # "OBV_ratio" → "obv_ratio", "_OBV" → "_obv", etc.
    obv_raw = _to_series(ta.obv(close=close, volume=volume), idx)
    pvt_raw = _to_series(ta.pvt(close=close, volume=volume), idx)

    obv_mean = obv_raw.rolling(20).mean().replace(0, np.nan)
    pvt_mean = pvt_raw.rolling(20).mean().replace(0, np.nan)

    features["obv_ratio"] = obv_raw / obv_mean
    features["pvt_ratio"] = pvt_raw / pvt_mean

    # Internal raw OBV: dùng cho rank cross-sectional (drop sau khi rank xong)
    # Giữ _obv để compute_rank_features có thể dùng nếu cần,
    # nhưng rank_map đã đổi sang obv_ratio (xem compute_rank_features).
    features["_obv"] = obv_raw
    features["_pvt"] = pvt_raw

    # ── RETURNS ───────────────────────────────────────────────────────────
    # pct_change(n) = backward-looking → không có lookahead bias
    # "RET_1D" → "ret_1d", etc.
    features["ret_1d"]  = close.pct_change(1)
    features["ret_5d"]  = close.pct_change(5)
    features["ret_10d"] = close.pct_change(10)

    # ── LAGGED FEATURES ──────────────────────────────────────────────────
    # shift(+n) = giá trị ngày hôm trước → backward-looking, không lookahead
    # "RSI_lag1" → "rsi_lag1", etc.
    features["rsi_lag1"]       = features["rsi"].shift(1)
    features["rsi_lag2"]       = features["rsi"].shift(2)
    features["macd_hist_lag1"] = features["macd_hist"].shift(1)
    features["rvi_lag1"]       = features["rvi"].shift(1)

    # ── ROLLING STATISTICS ───────────────────────────────────────────────
    # rolling(n).std() là backward-looking → không lookahead
    # "Close_rolling_std_20" → "close_rolling_std_20"
    # "Volume_ratio_20"      → "volume_ratio_20"
    features["close_rolling_std_20"] = close.pct_change(1).rolling(20).std()
    features["volume_ratio_20"]      = volume / volume.rolling(20).mean().replace(0, np.nan)

    return features


# ---------------------------------------------------------------------------
# Cross-sectional rank features
# rank_map keys → lowercase
# source columns cập nhật theo tên lowercase mới
# Dùng obv_ratio thay vì _obv raw cho OBV rank
# ---------------------------------------------------------------------------
def compute_rank_features(
    all_features: dict,
    all_data: dict,
    min_tickers: int = 15,
) -> dict:
    """
    Tính 6 cross-sectional rank features (percentile trong rổ VN30 mỗi ngày).

    Tên rank columns output đều lowercase:
        return_rank, atr_rank, rvi_rank, mfi_rank, obv_rank, volume_rank

    obv_rank dùng obv_ratio (OBV/rolling_mean, normalized,
    scale-free) thay vì _obv raw cumulative. OBV cumulative không scale-free:
    mã volume lớn hơn có OBV lớn hơn → rank không phản ánh momentum volume.

    Guard: nếu < min_tickers mã, trả về dict rỗng per ticker (CLAUDE.md §2).
    Rank chỉ tính trên common_dates (intersection) để mỗi ngày có đủ tất cả mã.
    """
    n_tickers = len(all_features)
    if n_tickers < min_tickers:
        logger.info(
            f"Rank features DISABLED: only {n_tickers} tickers < {min_tickers} required. "
            f"Cross-sectional features sẽ là NaN."
        )
        return {ticker: pd.DataFrame(index=f.index) for ticker, f in all_features.items()}

    logger.info(f"Computing rank features for {n_tickers} tickers...")

    common_dates = sorted(reduce(
        lambda a, b: a.intersection(b),
        [f.index for f in all_features.values()],
    ))

    if not common_dates:
        logger.warning("Rank features: no common dates across tickers — returning empty.")
        return {ticker: pd.DataFrame(index=f.index) for ticker, f in all_features.items()}

    tickers = list(all_features.keys())
    rank_features = {t: pd.DataFrame(index=common_dates) for t in tickers}

    # Keys lowercase
    # Source columns cập nhật theo tên lowercase mới
    # obv_rank dùng "obv_ratio" (normalized) thay vì "_obv" (raw cumulative)
    rank_map: dict[str, str | None] = {
        "return_rank": "ret_5d",     # 5-day return (backward-looking)
        "atr_rank":    "atr_pct",    # ATR % of price (scale-free)
        "rvi_rank":    "rvi",        # Relative Vigor Index
        "mfi_rank":    "mfi",        # Money Flow Index
        "obv_rank":    "obv_ratio",  # OBV/rolling_mean (scale-free, not raw)
        "volume_rank": None,         # raw volume từ OHLCV (cross-ticker so sánh hợp lý)
    }

    for rank_name, source_col in rank_map.items():
        matrix = pd.DataFrame(index=common_dates, columns=tickers, dtype=float)

        for t in tickers:
            if source_col is None:
                # Volume rank: dùng raw volume từ OHLCV (so sánh cross-ticker)
                col_data = all_data[t]["volume"].astype(float).reindex(common_dates)
            elif source_col in all_features[t].columns:
                col_data = all_features[t][source_col].reindex(common_dates)
            else:
                logger.debug(
                    f"  rank '{rank_name}': source col '{source_col}' "
                    f"not found for {t} — filling NaN."
                )
                col_data = pd.Series(np.nan, index=common_dates)
            matrix[t] = col_data

        # pct=True → percentile rank [0, 1]; na_option='keep' → NaN không tham gia rank
        ranked = matrix.rank(axis=1, pct=True, na_option="keep")
        for t in tickers:
            rank_features[t][rank_name] = ranked[t]

    logger.info(
        f"Rank features computed: {len(rank_map)} features × "
        f"{len(common_dates)} common dates."
    )
    return rank_features


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_feature_engineering(config: dict) -> dict:
    """
    Orchestrator chính cho Bước 2.

    Chiến lược dropna() [FIX-FE1]:
        Chỉ drop hàng NaN ở RAW feature columns (warm-up period của indicators).
        Rank feature columns được phép NaN ở ngày rìa — modeling layer xử lý.

    Ticker parsing [FIX-FE4]:
        Dùng filename_to_ticker() / ticker_to_filename() từ data_collection.

    Save strategy [WARNING-SAVE]:
        Atomic write per-ticker (tmp → rename) thay vì batch-delete-then-save.
        Không xoá toàn bộ file cũ trước — từng ticker được ghi đè an toàn.
    """
    paths    = config["paths"]
    feat_cfg = config["features"]

    ohlcv_dir = Path(paths["processed_data"]) / "ohlcv"
    feat_dir  = Path(paths["processed_data"]) / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    if not ohlcv_dir.exists():
        raise RuntimeError(f"OHLCV directory not found: {ohlcv_dir}. Run step 'data' first.")

    # ── Load OHLCV ───────────────────────────────────────────────────────
    # filename_to_ticker v3.1 raise ValueError — bọc try/except
    # Validate schema trước khi compute
    all_data: dict[str, pd.DataFrame] = {}
    for f in sorted(ohlcv_dir.glob("*.parquet")):
        try:
            ticker = filename_to_ticker(f.stem)
        except ValueError as e:
            logger.warning(f"Skipping unrecognized file '{f.name}': {e}")
            continue

        try:
            df = pd.read_parquet(f)
        except Exception as e:
            logger.warning(f"Cannot read '{f.name}': {e} — skipping.")
            continue

        # Validate schema — skip ticker nếu cột thiếu/sai dtype
        schema_errors = _validate_ohlcv(ticker, df)
        if schema_errors:
            logger.error(
                f"  [{ticker}] OHLCV schema invalid — skipping:\n"
                + "\n".join(f"    - {e}" for e in schema_errors)
            )
            continue

        all_data[ticker] = df

    if not all_data:
        raise RuntimeError(f"No valid OHLCV data found in {ohlcv_dir}. Run step 'data' first.")

    logger.info("=" * 60)
    logger.info(f"STEP 2: FEATURE ENGINEERING — {len(all_data)} tickers")
    logger.info("=" * 60)

    # ── Static features (TFT STATIC_REALS) — tính 1 lần trước vòng lặp ────
    # Lưu riêng vào static_features.parquet, không gộp vào time-series features.
    static_path = feat_dir / "static_features.parquet"
    try:
        static_df = compute_static_features(all_data)
        static_df.to_parquet(static_path, compression="zstd")
        logger.info(
            f"Static features saved: {len(static_df)} tickers -> {static_path}"
        )
    except Exception as e:
        logger.warning(f"compute_static_features failed: {e} — TFT market_cap_rank se thieu.")

    # ── Raw features (dropna() chưa gọi ở đây) ──────────────────────────
    all_features: dict[str, pd.DataFrame] = {}
    for ticker, df in all_data.items():
        logger.info(f"Computing raw features: {ticker}")
        try:
            raw_feat = compute_raw_features(df, feat_cfg)

            # Calendar features (TFT TIME_VARYING_KNOWN_REALS) — merge vào raw
            cal_feat = compute_calendar_features(df)
            all_features[ticker] = pd.concat([raw_feat, cal_feat], axis=1)
        except Exception as e:
            logger.error(f"  [{ticker}] compute_raw_features failed: {e} — skipping.")

    if not all_features:
        raise RuntimeError("All tickers failed raw feature computation.")

    # ── Rank features ────────────────────────────────────────────────────
    rank_features = compute_rank_features(
        all_features, all_data, feat_cfg.get("rank_min_tickers", 15)
    )

    # ── Merge rank vào raw features ──────────────────────────────────────
    for t in all_features:
        if t in rank_features and not rank_features[t].empty:
            all_features[t] = pd.concat([all_features[t], rank_features[t]], axis=1)

    # ── Save per-ticker với atomic write ─────────────────────────────────
    col_counts: list[int] = []
    saved = 0
    failed = 0

    for ticker, fdf in all_features.items():
        fdf = fdf.copy()

        # Drop internal columns (_obv, _pvt, _atr_raw)
        # Tên lowercase nhất quán với _INTERNAL_COLS constant
        fdf = fdf.drop(columns=[c for c in _INTERNAL_COLS if c in fdf.columns])

        # raw_cols_present xây dựng per-ticker (không dùng sample ticker đầu)
        # Loại rank cols (suffix "_rank"), internal cols, và calendar cols
        # Calendar features (dow_sin/cos, month_sin/cos, is_quarter_end) không có NaN
        # nên không cần trong subset — nhưng loại ra để tránh conflict nếu index có gap.
        _CALENDAR_COLS = frozenset({
            "dow_sin", "dow_cos", "month_sin", "month_cos", "is_quarter_end"
        })
        raw_cols_present = [
            c for c in fdf.columns
            if not c.endswith("_rank")
            and c not in _INTERNAL_COLS
            and c not in _CALENDAR_COLS
        ]

        before = len(fdf)

        # Drop hàng NaN ở raw feature cols — warm-up period indicators
        fdf = fdf.dropna(subset=raw_cols_present)
        dropped_raw = before - len(fdf)

        # Log rank NaN còn lại (monitoring — bình thường ở rìa series)
        rank_cols_present = [c for c in fdf.columns if c.endswith("_rank")]
        rank_nan_count = 0
        if rank_cols_present:
            rank_nan_count = int(fdf[rank_cols_present].isna().any(axis=1).sum())

        col_counts.append(len(fdf.columns))

        # Atomic write per-ticker (không xoá toàn bộ trước)
        ok = _save_feature_checkpoint(ticker, fdf, feat_dir)
        if ok:
            saved += 1
            logger.debug(
                f"  {ticker}: {before} → {len(fdf)} rows "
                f"(dropped {dropped_raw} warm-up NaN rows"
                + (f", {rank_nan_count} rows have rank NaN — OK" if rank_nan_count else "")
                + f") | {len(fdf.columns)} features"
            )
        else:
            failed += 1

    if not col_counts:
        raise RuntimeError("No tickers saved successfully.")

    logger.success(
        f"FEATURE ENGINEERING COMPLETED: "
        f"{saved} saved, {failed} failed | "
        f"Features per ticker → min={min(col_counts)}, max={max(col_counts)}, "
        f"avg={sum(col_counts)/len(col_counts):.1f} "
        f"(saved to {feat_dir})"
    )
    return all_features