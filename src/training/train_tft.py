"""
VN30 — Bước 4d: Temporal Fusion Transformer (TFT)

Theo CLAUDE.md §4d:
  - Implement SAU khi 4a, 4b, 4c hoạt động ổn định
  - Dùng pytorch-forecasting (pin version)
  - Chạy trên session riêng — không chạy cùng Ollama

Status: PLACEHOLDER — chưa implement.
"""

from loguru import logger


def run_tft_training(config):
    """
    TFT training — placeholder.
    
    Implement khi XGB/LGBM/LSTM/Stacking đã chạy ổn định.
    Yêu cầu:
        pytorch-forecasting==1.1.1
        pytorch-lightning==2.2.4
        torch==2.2.0
    """
    logger.warning(
        "TFT training chưa được implement. "
        "Đây là placeholder theo CLAUDE.md §4d. "
        "Implement sau khi 4a, 4b, 4c hoạt động ổn định."
    )
    return None
