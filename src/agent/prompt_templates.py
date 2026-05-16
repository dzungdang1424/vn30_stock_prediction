"""
Prompt templates cho LLM Agent — tiếng Việt.
Tách biệt khỏi logic để dễ chỉnh sửa nội dung mà không động vào code.
"""

SYSTEM_PROMPT = """Bạn là trợ lý phân tích kỹ thuật cổ phiếu thị trường Việt Nam (VN30).
Nhiệm vụ của bạn là đọc kết quả từ mô hình AI và giải thích bằng tiếng Việt tự nhiên, rõ ràng.

Nguyên tắc trả lời:
- Ngắn gọn, súc tích — không quá 150 từ
- Nêu rõ tín hiệu (MUA / BÁN / GIỮ), horizon (T+1, T+3), và xác suất
- Giải thích 1-2 lý do chính dựa trên các chỉ báo kỹ thuật được cung cấp nếu được hỏi
- LUÔN kết thúc bằng disclaimer
- Không bịa thêm thông tin ngoài dữ liệu được cung cấp"""


def build_prediction_prompt(ticker: str, data: dict) -> str:
    """
    Tạo prompt từ dữ liệu prediction JSON v2.1.
    """
    pred = data.get("predictions", {})
    
    direction_vi = {"UP": "TĂNG", "DOWN": "GIẢM"}
    signal_vi = {"BUY": "MUA", "SELL": "BÁN", "HOLD": "GIỮ", "NEUTRAL": "TRUNG LẬP"}

    prompt_parts = [f"Dữ liệu dự đoán từ mô hình AI cho mã {ticker}:"]
    
    for horizon in ["T+1", "T+3"]:
        if horizon in pred:
            h_data = pred[horizon]
            acc = h_data.get('model_accuracy', 0)
            
            prompt_parts.append(f"\nDự đoán {horizon}:")
            prompt_parts.append(f"  - Xu hướng: {direction_vi.get(h_data.get('direction', ''), h_data.get('direction', ''))}")
            prompt_parts.append(f"  - Xác suất: {h_data.get('probability', 0):.0%}")
            prompt_parts.append(f"  - Tín hiệu: {signal_vi.get(h_data.get('signal', ''), h_data.get('signal', ''))}")
            prompt_parts.append(f"  - Độ chính xác lịch sử: {acc:.0%}")
            
            features = h_data.get("top_features", [])
            if features:
                prompt_parts.append(f"  - Các yếu tố kỹ thuật ảnh hưởng nhất cho {horizon}:")
                for f in features:
                    prompt_parts.append(f"    + {f.get('name')}: {f.get('value')} (tầm quan trọng: {f.get('importance', 0):.0%})")

    prompt_parts.append(f"\nDữ liệu cập nhật đến: {data.get('data_date', 'N/A')}")
    prompt_parts.append("\nHãy giải thích kết quả trên bằng tiếng Việt tự nhiên, ngắn gọn.")

    return "\n".join(prompt_parts)


NOT_FOUND_MSG = (
    "Không tìm thấy mã '{ticker}' trong dữ liệu. "
    "Các mã hiện có: {available}.\n"
    "Gợi ý: thử nhập tên viết tắt như VCB, FPT, HPG..."
)

NO_DATA_MSG = (
    "Chưa có dữ liệu dự đoán cho {ticker}. "
    "Vui lòng chạy pipeline trước: python main.py --step train"
)

STALE_DATA_MSG = (
    "⚠️  Dữ liệu dự đoán cho {ticker} đã cũ ({days} ngày). "
    "Kết quả có thể không còn chính xác."
)
