"""
VN30 Stock Prediction — LLM Agent (Explanation Agent) v1.1

Đọc kết quả từ outputs/predictions/{ticker_filename}_latest.json
và trả lời câu hỏi dự đoán cổ phiếu bằng tiếng Việt.

Theo CLAUDE.md §6: đọc JSON → build prompt → gọi Gemini API → trả lời tự nhiên.

"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.agent.llm_client import LLMClient
from src.agent.prompt_templates import SYSTEM_PROMPT, build_prediction_prompt
from src.data_collection import ticker_to_filename


# ── Cấu hình ────────────────────────────────────────────────────────────────

# Disclaimer hardcode — CLAUDE.md §6: "hardcode constant, in sau mỗi response, không qua LLM"
DISCLAIMER = "\n⚠️  Đây là dự báo xác suất từ mô hình AI, không phải khuyến nghị đầu tư."

# Default predictions dir — overridable qua config (xem _get_predictions_dir)
_DEFAULT_PREDICTIONS_DIR = Path("outputs/predictions")

# Default LLM client — Gemini 2.5 Flash theo CLAUDE.md §3 & §6
_client = LLMClient(provider="gemini", model="gemini-2.5-flash")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_predictions_dir(config: dict | None) -> Path:
    """
    Đọc predictions dir từ config nếu có, fallback về default.
    Dùng config["paths"]["predictions_output"] — nhất quán với export_predictions.py.
    """
    if config:
        path_str = config.get("paths", {}).get("predictions_output")
        if path_str:
            return Path(path_str)
    return _DEFAULT_PREDICTIONS_DIR


# ── Core functions ───────────────────────────────────────────────────────────

def load_prediction(ticker: str, config: dict | None = None) -> dict | None:
    """
    Đọc file JSON prediction mới nhất cho ticker.

    Dùng ticker_to_filename() để build đúng tên file:
          "FPT.VN" → "FPT_VN_latest.json" (không phải "FPT.VN_latest.json").

    Validate data["ticker"] khớp ticker yêu cầu sau khi đọc,
          tránh giải thích nhầm mã khi file bị ghi đè sai.

    Returns dict nếu thành công, None nếu không có file hoặc lỗi.
    """
    pred_dir  = _get_predictions_dir(config)
    safe_name = ticker_to_filename(ticker)   # "FPT.VN" → "FPT_VN"
    path      = pred_dir / f"{safe_name}_latest.json"

    if not path.exists():
        logger.debug(f"Prediction file not found: {path}")
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Cannot parse prediction file '{path}': {e}")
        return None

    # Validate ticker nhất quán — phát hiện file bị ghi nhầm
    file_ticker = data.get("ticker", "")
    if file_ticker and file_ticker != ticker:
        logger.warning(
            f"Ticker mismatch: requested='{ticker}' but file contains ticker='{file_ticker}'. "
            f"File: {path.name}. Returning None để tránh giải thích sai mã."
        )
        return None

    return data


def check_staleness(data: dict) -> int:
    """
    Trả về số ngày kể từ lần generate (dùng để hiển thị thông tin cho user).

    Lưu ý: để kiểm tra prediction có stale không, dùng data["is_stale"] (bool)
    đã được tính chính xác trong export_predictions.py từ data_date.
    Hàm này chỉ tính "bao nhiêu ngày kể từ khi JSON được generate".
    """
    try:
        generated = datetime.fromisoformat(data["generated_at"])
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - generated
        return delta.days
    except Exception:
        return 0


def ask_llm(ticker: str, data: dict) -> str:
    """
    Build prompt từ JSON data và gọi LLM. Trả về raw LLM text (không có disclaimer).

    Kiểm tra predictions non-empty trước khi gọi LLM — tránh tốn quota
          khi JSON hợp lệ nhưng không có horizon nào được export.
    Bắt KeyError riêng để thông báo rõ khi thiếu GEMINI_API_KEY.

    Để có response đầy đủ kèm disclaimer, dùng explain_ticker() thay vì hàm này.
    """
    # Guard: predictions rỗng → không có gì để giải thích
    if not data.get("predictions"):
        logger.warning(f"[{ticker}] JSON hợp lệ nhưng predictions rỗng — bỏ qua gọi LLM.")
        return f"⚠️ Không có dự báo nào trong file prediction của {ticker}. Hãy chạy lại export."

    user_prompt = build_prediction_prompt(ticker, data)

    try:
        return _client.chat(user_prompt, system=SYSTEM_PROMPT)
    except KeyError as e:
        # Thường là thiếu GEMINI_API_KEY trong environment
        logger.error(f"Missing environment variable: {e}")
        return (
            f"⚠️ Thiếu biến môi trường {e}. "
            "Kiểm tra file .env và chạy lại với `python-dotenv`."
        )
    except TimeoutError as e:
        logger.error(f"LLM timeout: {e}")
        return f"⚠️ LLM không phản hồi. {e}"
    except ConnectionError as e:
        logger.error(f"LLM connection error: {e}")
        return f"⚠️ Không kết nối được LLM server. {e}"
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"⚠️ Lỗi khi gọi LLM: {e}"


def explain_ticker(ticker: str, config: dict | None = None) -> str:
    """
    Hàm duy nhất nên được CLI gọi để giải thích 1 ticker.

    Flow:
      1. Load prediction JSON (dùng ticker_to_filename để build đúng path).
      2. Kiểm tra is_stale từ JSON field (tính chính xác ở export_predictions).
      3. Gọi LLM.
      4. Append DISCLAIMER — luôn có, không phụ thuộc vào caller nhớ hay không.

    Returns:
      String hoàn chỉnh: LLM response + DISCLAIMER.
    """
    data = load_prediction(ticker, config)
    if data is None:
        return (
            f"⚠️ Không tìm thấy file prediction cho {ticker}. "
            "Hãy chạy export predictions trước."
        ) + DISCLAIMER

    # Staleness warning — dùng is_stale từ JSON (tính từ data_date, chính xác hơn generated_at)
    if data.get("is_stale", False):
        age_days = check_staleness(data)
        logger.warning(
            f"[{ticker}] Prediction stale (is_stale=True, generated ~{age_days} ngày trước). "
            "Kết quả có thể không phản ánh thị trường hiện tại."
        )

    answer = ask_llm(ticker, data)
    return answer + DISCLAIMER


def explain_batch(tickers: list[str], config: dict | None = None) -> dict:
    """
    Batch explain nhiều ticker — có time.sleep(4) để giữ dưới 15 RPM free tier.
    Theo CLAUDE.md §6: "thêm time.sleep(4) giữa các lần gọi".

    Gọi explain_ticker() thay vì ask_llm() trực tiếp — đảm bảo disclaimer
          luôn có trong mọi response, không cần caller tự append.
    """
    results = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = explain_ticker(ticker, config)
        # sleep sau mỗi ticker trừ cái cuối cùng
        if i < len(tickers) - 1:
            time.sleep(4)  # ~15 RPM safe buffer
    return results


def set_llm_client(provider: str = "gemini", model: str = "gemini-2.5-flash", **kwargs):
    """Cho phép runtime switch LLM provider (ví dụ: test với Ollama local)."""
    global _client
    _client = LLMClient(provider=provider, model=model, **kwargs)
    logger.info(f"LLM client switched to {provider}/{model}")