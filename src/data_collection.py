"""
VN30 Stock Prediction Pipeline — Bước 1: Thu Thập Dữ Liệu (v3.1)

"""

import os
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger


# ---------------------------------------------------------------------------
# Ticker ↔ filename helpers
# ---------------------------------------------------------------------------

# Bỏ ".HM" — không phải suffix chuẩn của HOSE (HOSE dùng ".VN")
# Thứ tự quan trọng: dài trước ngắn sau để tránh strip nhầm
_EXCHANGE_SUFFIXES = (".UPCOM", ".HNX", ".VN")


def _to_vnstock_symbol(ticker: str) -> str:
    """
    Chuyển ticker hệ thống → vnstock symbol (không suffix).
    Theo CLAUDE.md §4: FPT.VN → FPT

    Dùng _EXCHANGE_SUFFIXES (dài trước ngắn sau) để tránh strip nhầm.
    """
    ticker_upper = ticker.upper()
    for suffix in _EXCHANGE_SUFFIXES:
        if ticker_upper.endswith(suffix.upper()):
            return ticker[: -len(suffix)]
    return ticker


def _to_standard_ticker(symbol: str) -> str:
    """
    Chuyển vnstock symbol → ticker chuẩn hệ thống.
    Theo CLAUDE.md §4: FPT → FPT.VN
    """
    return f"{symbol}.VN"


def ticker_to_filename(ticker: str) -> str:
    """
    Chuyển ticker symbol → tên file (không có extension).

    Chỉ replace dấu "." của exchange suffix, KHÔNG replace dấu "." khác.

    Ví dụ:
        "VCB.VN"   → "VCB_VN"
        "HPG.VN"   → "HPG_VN"
        "VN30F1M"  → "VN30F1M"   (không có suffix → giữ nguyên)
    """
    for suffix in _EXCHANGE_SUFFIXES:
        if ticker.upper().endswith(suffix.upper()):
            base = ticker[: -len(suffix)]
            return base + suffix.replace(".", "_")
    return ticker


def filename_to_ticker(stem: str) -> str:
    """
    Chuyển file stem → ticker symbol (ngược với ticker_to_filename).

    Xóa suffix phụ trước ("_features", "_labels", "_ohlcv") rồi nhận diện exchange suffix.

    Ví dụ:
        "VCB_VN"           → "VCB.VN"
        "HPG_VN_features"  → "HPG.VN"
        "FPT_VN_ohlcv"     → "FPT.VN"

    Raise ValueError nếu không nhận ra exchange suffix
    thay vì fallback replace("_", ".") có thể trả về giá trị sai.
    """
    original_stem = stem

    for pipeline_suffix in ("_features", "_labels", "_ohlcv"):
        if stem.endswith(pipeline_suffix):
            stem = stem[: -len(pipeline_suffix)]
            break

    encoded_suffixes = [s.replace(".", "_") for s in _EXCHANGE_SUFFIXES]
    for enc, orig in zip(encoded_suffixes, _EXCHANGE_SUFFIXES):
        if stem.upper().endswith(enc.upper()):
            base = stem[: -len(enc)]
            return base + orig

    # Không fallback silent — raise rõ ràng để caller xử lý
    raise ValueError(
        f"filename_to_ticker: không nhận dạng exchange suffix trong '{original_stem}'. "
        f"Các suffix hợp lệ: {_EXCHANGE_SUFFIXES}. "
        f"Kiểm tra lại tên file hoặc cập nhật _EXCHANGE_SUFFIXES."
    )


# ---------------------------------------------------------------------------
# Download via vnstock (free tier — Quote class)
# NOTE: Đang dùng vnstock free. Khi migrate sang vnstock_data,
#       thay bằng Market().equity(symbol).ohlcv(start=..., end=...) theo CLAUDE.md §4.
# ---------------------------------------------------------------------------

def download_ohlcv(tickers, start_date="2022-01-01", end_date=None,
                   source="vci", retry_attempts=3, retry_delay=5):
    """
    Tải dữ liệu OHLCV từ vnstock (CLAUDE.md §4).

    Args:
        tickers:        list ticker chuẩn (VD: ["FPT.VN", "VCB.VN", ...])
        start_date:     ngày bắt đầu (YYYY-MM-DD)
        end_date:       ngày kết thúc (mặc định: hôm nay)
        source:         "vci" hoặc "kbs" (CLAUDE.md §4 — KHÔNG dùng TCBS).
                        Tự động normalize về lowercase.
        retry_attempts: số lần retry khi lỗi
        retry_delay:    delay giữa các lần retry (giây)

    Returns:
        dict[str, DataFrame]: {ticker: OHLCV DataFrame raw từ vnstock}
    """
    from vnstock import Quote

    # Normalize source về lowercase trước khi gọi API.
    # vnstock phân biệt hoa/thường; "VCI" có thể không được nhận diện.
    source_normalized = source.lower()
    if source_normalized not in ("vci", "kbs"):
        raise ValueError(
            f"Source '{source}' không được phép. CLAUDE.md §4: chỉ dùng 'vci' hoặc 'kbs'. "
            f"TCBS đã deprecated và bị cấm. Kiểm tra config.yaml key 'data.source'."
        )
    logger.info(f"Using data source: '{source_normalized}' (normalized from '{source}')")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # Số ngày giao dịch tối thiểu kỳ vọng: ~252 ngày/năm × số năm × 60%
    _start = datetime.strptime(start_date, "%Y-%m-%d")
    _end   = datetime.strptime(end_date,   "%Y-%m-%d")
    _years = max((_end - _start).days / 365.0, 0.5)
    min_rows = int(252 * _years * 0.6)
    logger.debug(f"  Minimum rows threshold: {min_rows} (based on {_years:.1f}yr window)")

    all_data = {}

    for i, ticker in enumerate(tqdm(tickers, desc="Downloading tickers")):
        symbol = _to_vnstock_symbol(ticker)
        # Sanity-check: symbol không được có suffix exchange còn sót
        if any(symbol.upper().endswith(s.upper()) for s in _EXCHANGE_SUFFIXES):
            logger.warning(
                f"  [{ticker}] _to_vnstock_symbol() vẫn còn suffix trong '{symbol}'. "
                f"API call có thể thất bại. Kiểm tra _EXCHANGE_SUFFIXES."
            )
        logger.info(f"Processing {ticker} (vnstock symbol: {symbol})")

        # Inter-ticker sleep để tránh rate limit (bắt đầu từ ticker thứ 2)
        if i > 0:
            time.sleep(2)

        df = None
        for attempt in range(1, retry_attempts + 1):
            try:
                quote = Quote(source=source_normalized, symbol=symbol)
                downloaded = quote.history(
                    start=start_date,
                    end=end_date,
                    interval="1D",
                )

                if downloaded is not None and len(downloaded) >= min_rows:
                    df = downloaded
                    logger.info(f"  -> Downloaded {len(df)} rows")
                    break
                else:
                    actual = len(downloaded) if downloaded is not None else 0
                    logger.warning(
                        f"  Attempt {attempt}: Insufficient data "
                        f"({actual} rows, need >= {min_rows})"
                    )
                    df = None
            except Exception as e:
                logger.warning(f"  Attempt {attempt} failed: {e}")
                df = None

            if attempt < retry_attempts:
                time.sleep(retry_delay)
            else:
                # Sleep sau attempt cuối cùng trước khi sang ticker tiếp.
                # Tránh cascade failure khi API đang rate-limit: không để vòng lặp
                # outer tiếp tục ngay lập tức sau nhiều lần retry liên tiếp.
                logger.debug(
                    f"  All {retry_attempts} attempts exhausted for {ticker}. "
                    f"Sleeping {retry_delay}s before next ticker."
                )
                time.sleep(retry_delay)

        if df is None or len(df) < min_rows:
            logger.error(f"  Failed to download {ticker} after {retry_attempts} attempts")
            continue

        all_data[ticker] = df

    logger.info(f"Downloaded successfully: {len(all_data)}/{len(tickers)} tickers")
    return all_data


# ---------------------------------------------------------------------------
# Schema normalization — vnstock output → OHLCV_SCHEMA (CLAUDE.md §4)
# ---------------------------------------------------------------------------

def normalize_schema(raw_data):
    """
    Chuẩn hóa output vnstock về OHLCV_SCHEMA.

    vnstock trả về: time, open, high, low, close, volume (columns, not index)
    CLAUDE.md yêu cầu:
        - Index: "date" (datetime64[ns])
        - Columns: open(f64), high(f64), low(f64), close(f64), volume(int64), ticker(str)
    """
    normalized = {}

    for ticker, df in raw_data.items():
        df = df.copy()

        # Nhận diện và chuẩn hóa cột date → set làm index.
        # Bao phủ các tên cột phổ biến từ các version vnstock khác nhau.
        _DATE_COL_CANDIDATES = ("time", "date", "Date", "datetime", "Datetime", "trading_date")
        date_col_found = None

        if not isinstance(df.index, pd.DatetimeIndex):
            # Date chưa là index — tìm trong columns
            for _cname in _DATE_COL_CANDIDATES:
                if _cname in df.columns:
                    date_col_found = _cname
                    break

            if date_col_found:
                df = df.rename(columns={date_col_found: "date"})
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                df = df.set_index("date")
            else:
                # Không tìm được cột date — log error và skip ticker này
                logger.error(
                    f"  [{ticker}] normalize_schema: không tìm được cột date. "
                    f"Columns hiện có: {list(df.columns)}. "
                    f"Ticker bị skip — kiểm tra output format của vnstock version đang dùng."
                )
                continue
        else:
            # Index đã là DatetimeIndex — chỉ cần normalize (strip time component)
            df.index = df.index.normalize()

        # Rename columns to lowercase nếu cần (vnstock 4.x đã lowercase)
        df.columns = df.columns.str.lower()
        df.index.name = "date"

        # Đảm bảo dtypes đúng
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype("float64")

        if "volume" in df.columns:
            # Chỉ cast dtype — KHÔNG fillna(0) ở đây.
            # check_quality() cần thấy NaN thật để tính missing ratio chính xác.
            # fillna(0) và cast int64 cuối cùng được thực hiện trong clean_data().
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")  # float64, NaN preserved

        # Thêm ticker column (CLAUDE.md bắt buộc)
        df["ticker"] = ticker

        # Kiểm tra cột bắt buộc — warn rõ và skip nếu thiếu (không drop im lặng)
        schema_cols = ["open", "high", "low", "close", "volume", "ticker"]
        _required_price_cols = ["open", "high", "low", "close", "volume"]
        missing_required = [c for c in _required_price_cols if c not in df.columns]
        if missing_required:
            logger.error(
                f"  [{ticker}] normalize_schema: thiếu cột bắt buộc {missing_required}. "
                f"Columns hiện có sau lowercase: {list(df.columns)}. "
                f"Ticker bị skip — kiểm tra output format của vnstock."
            )
            continue
        keep_cols = [c for c in schema_cols if c in df.columns]
        df = df[keep_cols]

        # Index dtype: datetime64[ns]
        df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns]")

        # Sort theo ngày
        df = df.sort_index()

        normalized[ticker] = df
        logger.debug(f"  {ticker}: schema normalized, {len(df)} rows")

    return normalized


# ---------------------------------------------------------------------------
# Quality check (CLAUDE.md §4) — phải gọi TRƯỚC clean_data
# Hàm này giờ được gọi trước khi ffill/dropna để
#                  missing ratio phản ánh dữ liệu thô thực tế từ API.
# ---------------------------------------------------------------------------

def check_quality(data, max_missing_ratio=0.20):
    """
    Kiểm tra chất lượng dữ liệu RAW (trước ffill/dropna) — CLAUDE.md §4:
      - Tỷ lệ missing < 20% — vượt → skip ticker
      - Không có giá trị âm
      - Không có consecutive unchanged close > 3 ngày (trừ khi có volume > 0)

    Phải gọi hàm này TRƯỚC clean_data. Nếu gọi sau ffill+dropna,
    missing ratio luôn = 0% và check hoàn toàn vô nghĩa.

    ffill consecutive check: loại trừ ngày có volume > 0
    (giá đứng thật do sàn/trần, không phải ffill artifact).
    """
    passed = {}
    price_cols = ["open", "high", "low", "close"]

    for ticker, df in data.items():
        issues = []

        avail = [c for c in price_cols if c in df.columns]

        # Tính missing ratio trên dữ liệu RAW (trước ffill).
        # Kiểm tra price cols + volume — volume missing nhiều cũng là dấu hiệu dữ liệu kém.
        check_cols = avail + (["volume"] if "volume" in df.columns else [])
        col_missing = df[check_cols].isna().mean()
        missing_ratio = col_missing.max()
        if missing_ratio > max_missing_ratio:
            issues.append(
                f"missing {missing_ratio:.1%} > {max_missing_ratio:.0%} "
                f"(worst col: {col_missing.idxmax()})"
            )

        # Giá trị âm (kiểm tra trên raw — trước khi bị mask/fill)
        for col in avail:
            neg = (df[col] < 0).sum()
            if neg > 0:
                issues.append(f"{col} has {neg} negative values")

        # ffill consecutive check với loại trừ volume > 0.
        # Nếu close không đổi nhưng volume > 0 → giao dịch thật (sàn/trần),
        # không phải ffill artifact → không flag.
        if "close" in df.columns:
            same_close = df["close"] == df["close"].shift(1)
            has_volume = (df.get("volume", pd.Series(0, index=df.index)) > 0)
            # Chỉ flag khi close không đổi VÀ không có volume (ffill artifact)
            suspected_ffill = same_close & ~has_volume
            group = (~suspected_ffill).cumsum()
            run_lengths = suspected_ffill.groupby(group).sum()
            mx = int(run_lengths.max()) if len(run_lengths) else 0
            if mx > 3:
                issues.append(
                    f"suspected ffill exceeds 3 days "
                    f"(max consecutive unchanged close + no volume = {mx})"
                )

        if issues:
            logger.warning(f"  {ticker} FAILED quality: {'; '.join(issues)}")
        else:
            passed[ticker] = df

    skipped = len(data) - len(passed)
    if skipped > 0:
        logger.warning(f"Quality check: {skipped} tickers skipped")
    logger.info(f"Tickers after quality check: {len(passed)}/{len(data)}")
    return passed


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(data):
    """
    Làm sạch dữ liệu OHLCV theo CLAUDE.md §4:
      - Loại giá âm/zero → NaN
      - Loại volume bất thường (>5x rolling median)
      - Xóa duplicate index
      - Forward-fill tối đa 3 ngày (CLAUDE.md bắt buộc)
      - Volume NaN → fill 0, cast int64
      - Drop rows thiếu giá

    Hàm này phải được gọi SAU check_quality.
           check_quality cần dữ liệu raw để tính missing ratio chính xác.
    """
    cleaned = {}
    price_cols = ["open", "high", "low", "close"]

    for ticker, df in data.items():
        logger.debug(f"Cleaning {ticker}...")
        df = df.copy()

        # Tách ticker column ra trước khi xử lý
        ticker_val = df["ticker"].iloc[0] if "ticker" in df.columns else ticker

        # Xóa duplicate index
        if df.index.duplicated().any():
            n_dup = df.index.duplicated().sum()
            df = df[~df.index.duplicated(keep="last")]
            logger.warning(f"  {ticker}: Removed {n_dup} duplicate date rows")

        available_price_cols = [c for c in price_cols if c in df.columns]

        # Tỷ lệ missing trước khi xử lý (log only — check thực đã làm ở check_quality)
        missing_before = df[available_price_cols].isna().mean()
        missing_summary = {c: f"{v:.1%}" for c, v in missing_before.items() if v > 0}
        if missing_summary:
            logger.info(f"  {ticker}: Missing ratio before cleaning: {missing_summary}")

        # Loại giá âm/zero
        for col in available_price_cols:
            df.loc[df[col] <= 0, col] = np.nan

        # Loại volume bất thường (>5x rolling median)
        if "volume" in df.columns:
            vol = df["volume"]
            # min_periods=10: cần ít nhất 10 điểm để median đủ ổn định.
            # min_periods=5 quá thấp — ở đầu chuỗi, median dễ bị chính outlier ảnh hưởng.
            vol_median = vol.rolling(20, min_periods=10).median()
            vol_spike_mask = (vol > (5 * vol_median)) & vol_median.notna()
            n_vol_outliers = vol_spike_mask.sum()
            if n_vol_outliers > 0:
                df.loc[vol_spike_mask, "volume"] = np.nan
                logger.warning(
                    f"  {ticker}: Detected {n_vol_outliers} volume outliers "
                    f"(>5x rolling median) — set to NaN"
                )

        # Drop all-NaN rows
        df = df.dropna(how="all", subset=available_price_cols)

        # Forward-fill tối đa 3 ngày (CLAUDE.md §4 bắt buộc)
        df[available_price_cols] = df[available_price_cols].ffill(limit=3)

        # Log gap lớn (nghỉ Tết ~7-10 ngày)
        if len(df) > 1:
            gaps = df.index.to_series().diff().dt.days
            large_gaps = gaps[gaps > 5]
            if not large_gaps.empty:
                logger.warning(
                    f"  {ticker}: Detected {len(large_gaps)} gaps >5 days "
                    f"(max={int(gaps.max())}d) — likely holidays."
                )

        # Volume NaN → fill 0, cast int64
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).astype("int64")

        # Drop rows thiếu giá
        df = df.dropna(subset=available_price_cols)
        df = df.sort_index()

        # Restore ticker column
        df["ticker"] = ticker_val

        cleaned[ticker] = df
        logger.debug(f"  -> {len(df)} rows after cleaning")

    return cleaned


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def filter_tickers(data, min_data_ratio=0.8):
    """Loại bỏ ticker có quá ít dữ liệu so với ticker có nhiều nhất."""
    if not data:
        return {}

    max_days = max(len(df) for df in data.values())
    threshold = int(max_days * min_data_ratio)

    filtered = {}
    for ticker, df in data.items():
        if len(df) >= threshold:
            filtered[ticker] = df
        else:
            logger.warning(f"Filtered out {ticker}: only {len(df)}/{max_days} days")

    logger.info(f"Tickers after filtering: {len(filtered)}/{len(data)}")
    return filtered


# ---------------------------------------------------------------------------
# Align dates
# ---------------------------------------------------------------------------

def align_dates(data, mode: str = "intersection"):
    """
    Căn chỉnh index ngày giao dịch giữa các ticker.

    mode="intersection": Chỉ giữ ngày có mặt ở TẤT CẢ ticker.
    mode="union": Giữ tất cả ngày, forward-fill limit=3 cho ngày thiếu.
    """
    if len(data) <= 1:
        return data

    if mode == "union":
        return _align_union(data)
    else:
        return _align_intersection(data)


def _align_intersection(data):
    common_dates = None
    for df in data.values():
        dates = set(df.index)
        common_dates = dates if common_dates is None else common_dates & dates

    common_dates = sorted(common_dates)

    if not common_dates:
        raise RuntimeError("No common trading dates found across tickers.")

    max_days = max(len(df) for df in data.values())
    dropped = max_days - len(common_dates)
    drop_ratio = dropped / max_days

    if drop_ratio > 0.05:
        logger.warning(
            f"Date alignment (intersection) dropped {dropped} days ({drop_ratio:.1%}). "
            f"Consider align_mode='union' in config."
        )

    logger.info(
        f"Common trading days: {len(common_dates)} "
        f"({common_dates[0].date()} -> {common_dates[-1].date()})"
    )

    aligned = {ticker: df.loc[common_dates].copy() for ticker, df in data.items()}
    return aligned


def _align_union(data):
    all_dates = sorted(set().union(*[df.index for df in data.values()]))
    max_days = max(len(df) for df in data.values())

    logger.info(
        f"Date alignment (union): {len(all_dates)} total days "
        f"({all_dates[0].date()} -> {all_dates[-1].date()})"
    )

    price_cols = ["open", "high", "low", "close"]
    aligned = {}
    for ticker, df in data.items():
        # Đánh dấu ngày nào đã có dữ liệu thật (trước reindex)
        original_dates = set(df.index)

        df_re = df.reindex(all_dates)

        # Forward-fill limit=3 CHỈ cho các ngày mới thêm vào qua reindex.
        # Các ngày đã có dữ liệu thật (original_dates) không bị ffill thêm.
        # Tránh cộng dồn: clean_data() đã ffill(limit=3) rồi, nếu align ffill tiếp
        # thì tổng có thể vượt 3 ngày (vi phạm CLAUDE.md §4).
        new_dates_mask = pd.Series(
            [d not in original_dates for d in df_re.index],
            index=df_re.index,
        )
        avail_prices = [c for c in price_cols if c in df_re.columns]

        # Thực hiện ffill trên toàn bộ series (limit=3), sau đó restore giá trị
        # của các ngày đã có dữ liệu thật (không cho phép ffill ghi đè chúng).
        original_vals = df_re[avail_prices].copy()
        df_re[avail_prices] = df_re[avail_prices].ffill(limit=3)
        # Restore ngày có dữ liệu gốc — chỉ để ffill fill vào ngày mới (NaN thật)
        df_re.loc[~new_dates_mask, avail_prices] = original_vals.loc[~new_dates_mask, avail_prices]

        # Volume fill 0
        if "volume" in df_re.columns:
            df_re["volume"] = df_re["volume"].fillna(0).astype("int64")

        # Ticker column: fill bằng hằng số ticker (không dùng bfill — backward-fill
        # không nhất quán với pattern fill của các cột khác trong pipeline).
        if "ticker" in df_re.columns:
            df_re["ticker"] = df_re["ticker"].ffill()
            # Nếu vẫn còn NaN ở đầu chuỗi (ngày trước ngày đầu tiên của ticker):
            # điền bằng ticker name trực tiếp thay vì bfill.
            if df_re["ticker"].isna().any():
                df_re["ticker"] = df_re["ticker"].fillna(ticker)

        # Drop rows still NaN in prices
        df_clean = df_re.dropna(subset=avail_prices)

        logger.debug(f"  {ticker}: {len(df)} -> {len(df_clean)} rows (union)")
        aligned[ticker] = df_clean

    return aligned


# ---------------------------------------------------------------------------
# Schema validation helper
# Dùng trong save_processed_data để validate trước khi lưu
# ---------------------------------------------------------------------------

_REQUIRED_SCHEMA = {
    "open":   "float64",
    "high":   "float64",
    "low":    "float64",
    "close":  "float64",
    "volume": "int64",
    "ticker": "object",  # str trong pandas là object dtype
}


def _validate_schema(ticker: str, df: pd.DataFrame) -> list[str]:
    """
    Kiểm tra schema trước khi lưu Parquet.
    Trả về danh sách lỗi (rỗng = pass).
    """
    errors = []

    # Index phải là DatetimeIndex tên "date"
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"Index is not DatetimeIndex (got {type(df.index).__name__})")
    if df.index.name != "date":
        errors.append(f"Index name is '{df.index.name}', expected 'date'")

    # Kiểm tra từng cột bắt buộc
    for col, expected_dtype in _REQUIRED_SCHEMA.items():
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")
        else:
            actual = str(df[col].dtype)
            if actual != expected_dtype:
                errors.append(
                    f"Column '{col}': dtype={actual}, expected={expected_dtype}"
                )

    # Không có giá âm
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                errors.append(f"Column '{col}' has {neg} negative values")

    return errors


# ---------------------------------------------------------------------------
# Save — per-ticker checkpoint
# Checkpoint: lưu parquet ngay sau khi từng ticker sẵn sàng.
# Atomic write: rename old → .bak trước, rename tmp → live,
#                  xóa .bak sau — tránh mất data nếu crash giữa chừng.
# Validate schema trước khi lưu.
# ---------------------------------------------------------------------------

def save_ticker_checkpoint(ticker: str, df: pd.DataFrame, ohlcv_dir: Path) -> bool:
    """
    Lưu một ticker ra Parquet ngay sau khi xử lý xong (checkpoint).
    Dùng atomic write: tmp file → rename, không mất dữ liệu nếu crash.

    Returns:
        True nếu lưu thành công, False nếu có lỗi schema.
    """
    # Validate schema trước khi lưu
    schema_errors = _validate_schema(ticker, df)
    if schema_errors:
        logger.error(
            f"  [{ticker}] Schema validation FAILED — skipping save:\n"
            + "\n".join(f"    - {e}" for e in schema_errors)
        )
        return False

    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{ticker_to_filename(ticker)}_ohlcv.parquet"
    live_path = ohlcv_dir / filename
    tmp_path  = ohlcv_dir / f"{filename}.tmp"
    bak_path  = ohlcv_dir / f"{filename}.bak"

    try:
        # 1. Lưu vào tmp trước
        df.to_parquet(tmp_path, compression="zstd")

        # 2. [FIX WARNING-1] Rename live → .bak (nếu tồn tại)
        if live_path.exists():
            live_path.rename(bak_path)

        # 3. Rename tmp → live
        tmp_path.rename(live_path)

        # 4. Xóa .bak sau khi rename thành công
        if bak_path.exists():
            bak_path.unlink()

        logger.success(f"  [{ticker}] Checkpoint saved: {live_path.name} ({len(df)} rows)")
        return True

    except Exception as e:
        logger.error(f"  [{ticker}] Failed to save checkpoint: {e}")
        # Rollback: nếu .bak còn đó thì restore lại
        if bak_path.exists() and not live_path.exists():
            bak_path.rename(live_path)
            logger.warning(f"  [{ticker}] Rolled back to backup.")
        # Xóa tmp lỡ bị kẹt
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def save_processed_data(data, processed_dir):
    """
    Lưu toàn bộ dữ liệu đã xử lý ra Parquet (batch save cuối pipeline).
    Dùng save_ticker_checkpoint cho từng ticker để đảm bảo atomic write.

    Hàm này vẫn được giữ cho compatibility với các bước sau (filter, align).
    Checkpoint per-ticker trong run_data_collection xử lý crash safety sớm hơn.
    """
    processed_dir = Path(processed_dir)
    saved = 0
    failed = 0

    for ticker, df in data.items():
        ok = save_ticker_checkpoint(ticker, df.copy(), processed_dir)
        if ok:
            saved += 1
        else:
            failed += 1

    logger.info(f"Batch save complete: {saved} saved, {failed} failed → {processed_dir}")
    if failed > 0:
        logger.warning(f"{failed} tickers NOT saved due to schema errors. Check logs above.")


# ---------------------------------------------------------------------------
# Merge helper
# Merge dữ liệu cũ + mới thay vì ghi đè
# ---------------------------------------------------------------------------

def merge_with_existing(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Gộp dữ liệu mới vào dữ liệu cũ.

    Chiến lược: concat → drop_duplicates(keep="last") → sort_index.
    Dữ liệu mới (new) được ưu tiên cho ngày trùng lặp (keep="last"
    vì new được đặt sau old trong concat).

    Args:
        existing: DataFrame cũ đã có trên disk
        new:      DataFrame vừa tải từ API (đã normalize schema)

    Returns:
        DataFrame merged, sorted by date
    """
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    logger.debug(
        f"  Merge: {len(existing)} (old) + {len(new)} (new) → {len(combined)} rows"
    )
    return combined


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_data_collection(config):
    """
    Main orchestrator cho Step 1.

    Thứ tự xử lý đã fix:
    1. Load existing data từ disk
    2. Quyết định ticker nào cần tải (incremental check)
    3. Download → normalize schema
    4. [FIX CRITICAL-3] check_quality TRƯỚC clean_data (missing ratio chính xác)
    5. clean_data (ffill, dropna)
    6. [FIX CRITICAL-2] merge_with_existing (concat, không ghi đè)
    7. [FIX CRITICAL-4] Checkpoint: lưu ngay sau mỗi ticker thành công
    8. filter_tickers → align_dates → batch save cuối
    """
    tickers = config["tickers"]
    data_cfg = config["data"]
    paths = config["paths"]

    logger.info("=" * 60)
    logger.info(f"STEP 1: DATA COLLECTION (vnstock) — {len(tickers)} tickers")
    logger.info("=" * 60)

    # Tính start_date từ period (VD: "3y" → 3 năm trước)
    period = data_cfg.get("period", "3y")
    years = int(period.replace("y", "")) if "y" in period else 3
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    ohlcv_dir = Path(paths["processed_data"]) / "ohlcv"

    # --- Load existing data ---
    existing_data: dict[str, pd.DataFrame] = {}
    if ohlcv_dir.exists():
        for f in ohlcv_dir.glob("*.parquet"):
            # filename_to_ticker giờ raise ValueError thay vì silent
            try:
                ticker = filename_to_ticker(f.stem)
            except ValueError as e:
                logger.warning(f"Skipping unrecognized file '{f.name}': {e}")
                continue
            try:
                df = pd.read_parquet(f)
                existing_data[ticker] = df
                logger.debug(f"Loaded existing: {ticker} ({len(df)} rows)")
            except Exception as e:
                logger.warning(f"Could not load existing data {f.name}: {e}")

    # --- Incremental check ---
    # Dùng ngưỡng 4 ngày calendar để tính business-day gap.
    # Thứ 2 so với thứ 6 = 3 ngày calendar nhưng chỉ 1 phiên giao dịch.
    # Ngưỡng 4 ngày bao phủ được: thứ 6 → thứ 2 (3 ngày) và ngày nghỉ lễ 1 ngày.
    # Nếu cần chính xác hơn, dùng pandas_market_calendars (optional dependency).
    INCREMENTAL_THRESHOLD_DAYS = 4

    tickers_to_download = []
    for t in tickers:
        if t in existing_data and not existing_data[t].empty:
            last_date = existing_data[t].index.max()
            # Timezone-safe: normalize cả hai về naive datetime trước khi trừ.
            # last_date từ Parquet có thể có timezone (tz-aware Timestamp),
            # datetime.now() là naive — trừ trực tiếp sẽ raise TypeError.
            try:
                last_date_naive = last_date.to_pydatetime()
                if last_date_naive.tzinfo is not None:
                    last_date_naive = last_date_naive.replace(tzinfo=None)
                days_diff = (datetime.now() - last_date_naive).days
            except Exception as e:
                logger.warning(
                    f"  [{t}] Không thể tính days_diff cho incremental check: {e}. "
                    f"Sẽ tải lại dữ liệu để an toàn."
                )
                days_diff = INCREMENTAL_THRESHOLD_DAYS + 1  # force download

            if days_diff <= INCREMENTAL_THRESHOLD_DAYS:
                logger.info(
                    f"[{t}] Up-to-date (last: {last_date.date()}, "
                    f"{days_diff}d ago <= {INCREMENTAL_THRESHOLD_DAYS}d threshold). Skipping."
                )
                continue
        tickers_to_download.append(t)

    logger.info(
        f"Incremental check: {len(tickers) - len(tickers_to_download)} up-to-date, "
        f"{len(tickers_to_download)} need download."
    )

    # combined_data: sẽ chứa kết quả cuối cùng (existing + newly merged)
    combined_data: dict[str, pd.DataFrame] = existing_data.copy()

    if tickers_to_download:
        raw_data = download_ohlcv(
            tickers=tickers_to_download,
            start_date=start_date,
            end_date=end_date,
            # normalize_schema trong download_ohlcv đã xử lý lowercase
            source=data_cfg.get("source", "vci"),
            retry_attempts=data_cfg.get("retry_attempts", 3),
            retry_delay=data_cfg.get("retry_delay_seconds", 5),
        )

        if raw_data:
            # Schema normalization: vnstock output → OHLCV_SCHEMA
            normalized = normalize_schema(raw_data)

            # check_quality TRƯỚC clean_data
            # Missing ratio phải được tính trên dữ liệu raw, trước khi ffill/dropna
            quality_passed = check_quality(normalized, data_cfg.get("max_missing_ratio", 0.20))

            # clean_data sau quality check
            cleaned = clean_data(quality_passed)

            # Merge với existing thay vì ghi đè
            # Checkpoint: lưu ngay sau mỗi ticker
            for t, new_df in cleaned.items():
                if t in existing_data and not existing_data[t].empty:
                    merged_df = merge_with_existing(existing_data[t], new_df)
                else:
                    merged_df = new_df
                    logger.debug(f"  [{t}] No existing data — using downloaded data as-is.")

                combined_data[t] = merged_df

                # Checkpoint lưu ngay sau mỗi ticker thành công
                save_ticker_checkpoint(t, merged_df.copy(), ohlcv_dir)

        else:
            logger.warning("No new data downloaded in this run. Existing data preserved.")
    else:
        logger.info("All tickers are up-to-date. No downloads needed.")

    if not combined_data:
        raise RuntimeError("No data available (both existing and new downloads are empty).")

    # Filter (min data ratio)
    filtered = filter_tickers(combined_data, data_cfg.get("min_data_ratio", 0.8))

    if not filtered:
        raise RuntimeError("All tickers were filtered out.")

    # Align dates
    align_mode = data_cfg.get("align_mode", "intersection")
    aligned = align_dates(filtered, mode=align_mode)

    # Batch save cuối (sau align — index có thể thay đổi sau intersection/union).
    # Cảnh báo: align_dates(mode="intersection") có thể drop rows so với checkpoint
    # per-ticker đã lưu ở bước trên. Đây là behavior đúng (aligned data nhất quán
    # để train cross-sectional features), nhưng cần log rõ để không gây bất ngờ.
    if align_mode == "intersection":
        for t, aligned_df in aligned.items():
            pre_align_rows = len(combined_data.get(t, pd.DataFrame()))
            if len(aligned_df) < pre_align_rows:
                logger.info(
                    f"  [{t}] align_dates(intersection) giảm {pre_align_rows} → "
                    f"{len(aligned_df)} rows ({pre_align_rows - len(aligned_df)} ngày bị drop). "
                    f"File checkpoint sẽ được cập nhật với aligned data."
                )
    save_processed_data(aligned, ohlcv_dir)

    logger.success(f"DATA COLLECTION COMPLETED: {len(aligned)} tickers")
    return aligned