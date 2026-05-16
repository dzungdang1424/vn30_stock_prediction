"""
VN30 Stock Prediction — Ticker Resolver

Tìm mã cổ phiếu chuẩn (FPT.VN) từ câu hỏi tự nhiên của người dùng.
Dùng fuzzy matching trên ticker_map.json.
"""

import json
import re
from pathlib import Path

from loguru import logger


# Load ticker map một lần khi import module
_MAP_PATH = Path(__file__).parent / "ticker_map.json"

if _MAP_PATH.exists():
    with open(_MAP_PATH, "r", encoding="utf-8") as f:
        TICKER_MAP: dict[str, str] = json.load(f)
else:
    logger.warning(f"ticker_map.json not found at {_MAP_PATH}")
    TICKER_MAP = {}

# Pre-compute upper_map và all_tickers (không rebuild mỗi lần gọi)
_UPPER_MAP: dict[str, str] = {k.upper(): v for k, v in TICKER_MAP.items()}
_ALL_TICKERS: set[str] = set(TICKER_MAP.values())


def resolve_ticker(user_input: str) -> str | None:
    """
    Tìm ticker chuẩn (e.g. "FPT.VN") từ input tự do của người dùng.
    Hỗ trợ trích xuất từ câu tự nhiên (VD: "Phân tích mã FPT ngày mai").
    """
    normalized = user_input.strip().upper()

    # 1. Trích xuất các từ (có thể chứa dấu chấm như FPT.VN)
    words = re.findall(r'[A-Z0-9\.]+', normalized)

    # Ưu tiên kiểm tra các mã 3 chữ cái đứng độc lập (FPT, VCB,...)
    for word in words:
        if word in _ALL_TICKERS:
            return word
        if f"{word}.VN" in _ALL_TICKERS:
            return f"{word}.VN"
        if word in _UPPER_MAP:
            return _UPPER_MAP[word]

    # 2. Tìm kiếm tên đầy đủ (phrase) trong câu nếu không thấy mã ngắn
    for key in sorted(_UPPER_MAP.keys(), key=len, reverse=True):
        if len(key) >= 3 and key in normalized:
            return _UPPER_MAP[key]

    return None


def get_popular_tickers(n: int = 5) -> list[str]:
    """Trả về n mã phổ biến nhất để gợi ý khi không nhận ra ticker."""
    popular = ["FPT.VN", "VCB.VN", "HPG.VN", "VNM.VN", "VIC.VN"]
    return popular[:n]
