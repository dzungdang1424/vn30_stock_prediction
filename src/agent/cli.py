"""
VN30 Stock Prediction — CLI Interface

Entry point: vòng lặp hỏi-đáp cho LLM Agent.
Luồng theo CLAUDE.md §6:
    1. User nhập câu hỏi
    2. ticker_resolver → tìm ticker (fuzzy match từ ticker_map.json)
    3. Kiểm tra outputs/predictions/{ticker}_latest.json tồn tại
    4. Kiểm tra is_stale → cảnh báo nếu dữ liệu cũ
    5. llm_agent → build prompt + gọi LLM
    6. In response + DISCLAIMER (hardcode, không qua LLM)
    7. Loop
"""

import sys
import glob
from pathlib import Path

from loguru import logger

from src.agent.ticker_resolver import resolve_ticker, get_popular_tickers
from src.agent.llm_agent import load_prediction, check_staleness, ask_llm
from src.agent.prompt_templates import (
    NOT_FOUND_MSG,
    NO_DATA_MSG,
    STALE_DATA_MSG,
)

# Disclaimer hardcode — luôn append, không exception, không qua LLM
DISCLAIMER = "\n Đây là dự báo xác suất từ mô hình AI, không phải khuyến nghị đầu tư."

# Cấu hình mặc định
PREDICTIONS_DIR = Path("outputs/predictions")
STALE_THRESHOLD_DAYS = 2


def list_available_tickers() -> list[str]:
    """Liệt kê các ticker đã có file prediction."""
    files = glob.glob(str(PREDICTIONS_DIR / "*_latest.json"))
    return sorted(Path(f).stem.replace("_latest", "") for f in files)


def answer(user_input: str) -> str:
    """
    Hàm chính: nhận câu hỏi tự do → trả về câu trả lời + disclaimer.
    Dùng để tích hợp vào bất kỳ interface nào.
    """
    # Step 2: Ticker resolver
    ticker = resolve_ticker(user_input)

    if not ticker:
        # Gợi ý 5 mã phổ biến + danh sách có sẵn
        available = list_available_tickers()
        popular = get_popular_tickers(5)
        return NOT_FOUND_MSG.format(
            ticker=user_input,
            available=", ".join(t.replace(".VN", "") for t in available) if available else ", ".join(t.replace(".VN", "") for t in popular)
        ) + DISCLAIMER

    # Step 3: Kiểm tra JSON tồn tại
    data = load_prediction(ticker)
    if not data:
        return NO_DATA_MSG.format(ticker=ticker) + DISCLAIMER

    # Step 4: Kiểm tra is_stale
    stale_days = check_staleness(data)
    stale_warning = ""
    if stale_days >= STALE_THRESHOLD_DAYS:
        stale_warning = STALE_DATA_MSG.format(ticker=ticker, days=stale_days) + "\n\n"

    # Step 5: LLM Agent
    answer_text = ask_llm(ticker, data)

    # Step 6: Response + DISCLAIMER (hardcode)
    return stale_warning + answer_text + DISCLAIMER


def run_cli(initial_ticker: str | None = None) -> None:
    """Interactive CLI loop."""
    available = list_available_tickers()
    if not available:
        print("❌ Chưa có dữ liệu dự đoán. Chạy: make run-pipeline")
        sys.exit(1)

    print("=" * 55)
    print("  VN30 Stock Prediction — AI Assistant")
    print("=" * 55)
    print(f"  Dữ liệu có sẵn: {', '.join(t.replace('.VN', '') for t in available)}")
    print("  Gõ 'exit' hoặc Ctrl+C để thoát.")
    print("=" * 55)

    # Nếu truyền --ticker thẳng vào, hỏi 1 lần rồi thoát
    if initial_ticker:
        print(f"\n🔍 Đang phân tích {initial_ticker.upper()}...\n")
        print(answer(initial_ticker))
        return

    # Step 7: Loop
    while True:
        try:
            user_input = input("\nBạn hỏi: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTạm biệt!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "thoát"):
            print("Tạm biệt!")
            break

        print("\n🤖 AI đang phân tích...\n")
        print(answer(user_input))
