"""
VN30 Stock Prediction Pipeline — Bước 3: Label Engineering (v2.1)

Tạo continuous future returns T+1 và T+3. Binary labeling bị defer hoàn toàn
sang Bước 4 (Modeling) bên trong mỗi Walk-Forward fold.

"""

from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from loguru import logger

from src.data_collection import filename_to_ticker, ticker_to_filename


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API của module này (những gì training layer được phép import):
#
#   compute_dynamic_threshold()  — tính threshold rolling, không binary label
#   apply_purge_embargo()        — áp purge/embargo per fold boundary
#   log_label_quality()          — log stats sau khi binarize (training layer)
#   run_label_engineering()      — orchestrator Bước 3
#
# KHÔNG import create_labels() hay select_multiplier() từ đây —
# chúng đã bị xoá khỏi module. Xem ARCHITECTURE NOTE ở docstring.
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Threshold helper (safe to export — không tạo binary label)
# ---------------------------------------------------------------------------

def compute_dynamic_threshold(
    close: pd.Series,
    window: int = 20,
    multiplier: float = 1.0,
) -> pd.Series:
    """
    Tính ngưỡng động = multiplier × rolling_std(return_1d, window).

    Chỉ dùng thông tin quá khứ (rolling backward) → không có lookahead bias.
    An toàn để export và dùng ở bất kỳ bước nào.

    Args:
        close:      Series giá close (float64, DatetimeIndex)
        window:     Rolling window để tính std (mặc định 20 ngày)
        multiplier: Hệ số nhân (chọn per-fold ở training layer)

    Returns:
        Series threshold cùng index với close.
    """
    returns = close.pct_change(1)
    threshold = multiplier * returns.rolling(window).std()
    return threshold


# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Purging & Embargo (safe to export — chỉ set NaN, không tạo label mới)
# ---------------------------------------------------------------------------

def apply_purge_embargo(
    labels: pd.Series,
    fold_boundaries: list,
    horizon: int,
    embargo_days: int = 0,
) -> pd.Series:
    """
    Áp dụng Purging và Embargo tại các ranh giới train/test.

    Hàm này nhận labels bất kỳ (continuous float hoặc binary 0/1/NaN) và
    chỉ set một số vị trí thành NaN — không tạo label mới. An toàn để
    export và gọi từ training layer sau khi có fold_boundaries.

    Purging:  Set NaN cho `horizon` ngày giao dịch CUỐI mỗi train window.
              Label tại ngày t dùng giá t+horizon — nếu t gần sát test_start,
              label đó "nhìn vào" test set → phải xoá.

    Embargo:  Set NaN cho `embargo_days` ngày giao dịch ĐẦU mỗi test window.
              Tránh feature windows (VD: rolling 20 ngày) bị lẫn thông tin train.
              Với horizon <= 3, embargo_days = horizon là đủ. Tắt: embargo_days=0.

    Args:
        labels:          Series labels (continuous hoặc binary, có thể chứa NaN)
        fold_boundaries: list of (train_end, test_start) — datetime objects
                         VD: [(fold.train_end, fold.test_start) for fold in folds]
        horizon:         Label horizon tính theo ngày GIAO DỊCH (không phải calendar)
        embargo_days:    Số ngày giao dịch embargo sau test_start (mặc định = 0)

    Returns:
        Series labels mới với vùng purge/embargo = NaN.

    Dùng searchsorted trên index thực tế thay vì pd.Timedelta —
    đảm bảo đúng horizon ngày GIAO DỊCH bất kể lịch nghỉ cuối tuần / Tết VN.

    NOTE: Khuyến nghị gọi ở modeling layer khi đã có danh sách folds đầy đủ,
    không gọi từ run_label_engineering() vì lúc đó chưa có fold_boundaries.
    """
    labels = labels.copy()
    idx = labels.index

    for train_end, test_start in fold_boundaries:
        # ── Purging ──────────────────────────────────────────────────────
        # searchsorted('right') → vị trí ngay SAU train_end trong index.
        # Purge `horizon` ngày giao dịch tính ngược về từ vị trí đó.
        train_end_pos = idx.searchsorted(train_end, side="right")
        purge_start_pos = max(0, train_end_pos - horizon)
        purge_idx = idx[purge_start_pos:train_end_pos]
        if len(purge_idx):
            labels.loc[purge_idx] = np.nan
            logger.debug(
                f"  Purged {len(purge_idx)} trading days "
                f"({purge_idx[0].date()} → {purge_idx[-1].date()})"
            )

        # ── Embargo ──────────────────────────────────────────────────────
        if embargo_days > 0:
            test_start_pos = idx.searchsorted(test_start, side="left")
            embargo_end_pos = min(len(idx), test_start_pos + embargo_days)
            embargo_idx = idx[test_start_pos:embargo_end_pos]
            if len(embargo_idx):
                labels.loc[embargo_idx] = np.nan
                logger.debug(
                    f"  Embargo {len(embargo_idx)} trading days "
                    f"({embargo_idx[0].date()} → {embargo_idx[-1].date()})"
                )

    return labels


# ---------------------------------------------------------------------------
# Quality logger (chỉ hợp lệ với BINARY labels từ training layer)
# ---------------------------------------------------------------------------

def log_label_quality(
    labels: pd.Series,
    horizon: int,
    total_rows: int = None,
) -> None:
    """
    Log thống kê chất lượng binary labels sau khi binarize per-fold.

    ⚠️  QUAN TRỌNG — Input type:
        Hàm này chỉ hợp lệ với BINARY labels (values ∈ {0, 1, NaN}).
        KHÔNG gọi hàm này với continuous return Series (future_return_t1/t3)
        từ run_label_engineering() — kết quả sẽ sai hoàn toàn vì
        (valid == 1).mean() kiểm tra giá trị == 1.0 chính xác, không phải
        "positive return".

        Gọi đúng chỗ: training layer, SAU khi create_labels() đã chạy
        bên trong Walk-Forward fold.

    Báo cáo neutral zone ratio (|return| < threshold bị NaN)
    và cảnh báo nếu > 30% — thị trường sideways có thể làm dataset biased.

    Args:
        labels:     Binary Series (0/1/NaN). NaN = neutral zone hoặc warm-up.
        horizon:    Label horizon (để log rõ T+1 hay T+3)
        total_rows: Tổng số hàng gốc trước khi tạo label. Dùng để ước tính
                    neutral zone chính xác hơn (loại trừ warm-up NaN ở đầu).
                    Nếu None, dùng len(labels).

    Thêm guard kiểm tra input không phải continuous float.
    """
    # Guard: cảnh báo nếu labels trông như continuous returns
    # (có giá trị ngoài {0, 1, NaN} — continuous float thường có nhiều giá trị như vậy)
    non_binary_mask = labels.dropna().apply(lambda x: x not in (0, 1, 0.0, 1.0))
    if non_binary_mask.any():
        n_bad = non_binary_mask.sum()
        logger.warning(
            f"  log_label_quality T+{horizon}: {n_bad} non-binary values detected "
            f"(e.g. {labels.dropna()[non_binary_mask].iloc[0]:.6f}). "
            f"Hàm này chỉ hợp lệ với binary labels (0/1/NaN). "
            f"Đừng gọi với continuous future_return Series."
        )
        return

    total     = len(labels)
    nan_count = int(labels.isna().sum())
    valid     = labels.dropna()

    pct_pos   = float((valid == 1).mean() * 100) if len(valid) > 0 else 0.0
    pct_neg   = float((valid == 0).mean() * 100) if len(valid) > 0 else 0.0
    imbalance = abs(pct_pos - pct_neg)

    base = total_rows if total_rows is not None else total
    neutral_ratio = nan_count / base if base > 0 else 0.0

    logger.info(
        f"  T+{horizon}: {total} rows | "
        f"valid={len(valid)} ({len(valid)/total*100:.1f}%) | "
        f"neutral/NaN={nan_count} ({neutral_ratio*100:.1f}%) | "
        f"↑{pct_pos:.1f}%  ↓{pct_neg:.1f}% | "
        f"imbalance={imbalance:.1f}pp"
    )

    if neutral_ratio > 0.30:
        logger.warning(
            f"  T+{horizon}: neutral zone cao ({neutral_ratio*100:.1f}% > 30%) — "
            f"thị trường sideways nhiều, dataset chỉ chứa ngày biến động mạnh. "
            f"Xem xét giảm threshold_multiplier trong config."
        )
    if imbalance > 20:
        logger.warning(
            f"  T+{horizon}: class imbalance {imbalance:.1f}pp — "
            f"consider class_weight='balanced' in models."
        )


# ---------------------------------------------------------------------------
# Save helper — atomic write per-ticker (checkpoint)
# Lưu checkpoint ngay sau mỗi ticker thay vì batch cuối
# ---------------------------------------------------------------------------

def _save_label_checkpoint(
    ticker: str,
    ldf: pd.DataFrame,
    label_dir: Path,
) -> bool:
    """
    Lưu label DataFrame ra Parquet với atomic write (tmp → rename).

    Atomic write: ghi ra .tmp trước, rename .live → .bak, rename .tmp → live,
    xoá .bak. Nếu crash giữa chừng, .bak còn đó để rollback thủ công.

    Returns:
        True nếu lưu thành công, False nếu lỗi.
    """
    label_dir.mkdir(parents=True, exist_ok=True)

    filename  = f"{ticker_to_filename(ticker)}_labels.parquet"
    live_path = label_dir / filename
    tmp_path  = label_dir / f"{filename}.tmp"
    bak_path  = label_dir / f"{filename}.bak"

    try:
        ldf.to_parquet(tmp_path, compression="zstd")

        if live_path.exists():
            live_path.rename(bak_path)

        tmp_path.rename(live_path)

        if bak_path.exists():
            bak_path.unlink()

        logger.debug(f"  Checkpoint saved: {live_path.name} ({len(ldf)} rows)")
        return True

    except Exception as e:
        logger.error(f"  [{ticker}] Failed to save label checkpoint: {e}")
        # Rollback nếu .bak còn đó và live đã bị rename đi
        if bak_path.exists() and not live_path.exists():
            bak_path.rename(live_path)
            logger.warning(f"  [{ticker}] Rolled back to backup.")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_label_engineering(config: dict) -> dict:
    """
    Orchestrate Bước 3: Tính continuous future returns cho tất cả tickers.

    Per CLAUDE.md §3: Bước này CHỈ tính tỷ suất sinh lời tương lai.
    Binary labeling (0/1) được defer hoàn toàn sang Bước 4 (Modeling),
    chạy ĐỘNG bên trong mỗi Walk-Forward fold.

    Output columns mỗi ticker:
        - close:              float64 — training dùng để tính threshold per-fold
        - future_return_t1:   float64 — close.shift(-1)/close - 1
        - future_return_t3:   float64 — close.shift(-3)/close - 1
        (hoặc horizon khác theo config["labels"]["horizons"])

    Returns:
        dict[ticker → DataFrame] với continuous returns đã lưu ra Parquet.
    """
    paths         = config["paths"]
    label_cfg     = config["labels"]
    processed_dir = Path(paths["processed_data"])
    horizons: list[int] = label_cfg["horizons"]

    logger.info("=" * 60)
    logger.info(f"STEP 3: LABEL ENGINEERING — horizons={horizons} (continuous returns only)")
    logger.info("=" * 60)

    # ── Load OHLCV ───────────────────────────────────────────────────────
    ohlcv_dir = processed_dir / "ohlcv"
    if not ohlcv_dir.exists():
        raise RuntimeError(
            f"OHLCV directory not found: {ohlcv_dir}. Run step 'data' first."
        )

    all_data: dict[str, pd.DataFrame] = {}
    for f in sorted(ohlcv_dir.glob("*.parquet")):
        # filename_to_ticker v3.1 raise ValueError — bọc try/except
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

        # Validate và cast dtype của cột 'close' trước khi dùng
        if "close" not in df.columns:
            logger.error(f"  {ticker}: 'close' column missing in '{f.name}' — skipping.")
            continue

        if not pd.api.types.is_float_dtype(df["close"]):
            logger.warning(
                f"  {ticker}: 'close' dtype={df['close'].dtype}, expected float64. "
                f"Casting to float64."
            )
            try:
                df["close"] = df["close"].astype("float64")
            except (ValueError, TypeError) as e:
                logger.error(
                    f"  {ticker}: Cannot cast 'close' to float64: {e} — skipping."
                )
                continue

        all_data[ticker] = df

    if not all_data:
        raise RuntimeError(
            f"No valid OHLCV parquet files in {ohlcv_dir}. Run step 'data' first."
        )

    logger.info(f"Loaded {len(all_data)} tickers from {ohlcv_dir}")

    # ── Compute continuous returns ────────────────────────────────────────
    label_dir = processed_dir / "labels"
    all_labels: dict[str, pd.DataFrame] = {}

    for ticker, df in all_data.items():
        logger.info(f"Computing future returns for {ticker}...")

        close = df["close"]
        label_df = pd.DataFrame(index=df.index)

        # Lưu close để training step dùng cho select_multiplier per-fold
        label_df["close"] = close

        for h in horizons:
            # Continuous target ONLY — per CLAUDE.md §3
            # shift(-h) ở đây là ĐÚNG: tính return tương lai tại ngày t
            future_return = close.shift(-h) / close - 1
            label_df[f"future_return_t{h}"] = future_return

            valid_count = int(future_return.notna().sum())
            nan_count   = int(future_return.isna().sum())
            logger.info(
                f"  T+{h}: {valid_count} valid returns, "
                f"{nan_count} NaN (last {h} rows of series — expected)"
            )

        # Đổi how="all" → how="any":
        # Đảm bảo mọi horizon đều có continuous return trước khi lưu.
        # how="all" giữ lại rows chỉ một horizon có giá trị — không nhất quán
        # khi training join label T+1 và T+3 cùng lúc.
        return_cols = [f"future_return_t{h}" for h in horizons]
        rows_before = len(label_df)
        label_df = label_df.dropna(subset=return_cols, how="any")
        rows_dropped = rows_before - len(label_df)

        if rows_dropped > 0:
            logger.info(
                f"  Dropped {rows_dropped} rows (missing ≥1 horizon return) — "
                f"last {max(horizons)} rows of series. {len(label_df)} rows remain."
            )

        all_labels[ticker] = label_df

        # Checkpoint: lưu ngay sau mỗi ticker thành công
        _save_label_checkpoint(ticker, label_df.copy(), label_dir)

    logger.success(
        f"LABEL ENGINEERING COMPLETED: {len(all_labels)} tickers → {label_dir}\n"
        f"Output columns: close + {return_cols}\n"
        f"Binary labeling deferred to Walk-Forward folds in Step 4 (CLAUDE.md §3)."
    )
    return all_labels