"""
VN30 Stock Prediction Pipeline — Main Orchestrator

Entry point theo CLAUDE.md §6:
    python main.py --mode pipeline --base-path ./
    python main.py --mode chat --base-path ./
    python main.py --mode train-tft --base-path ./

Hoặc chạy từng bước:
    python main.py --step data
    python main.py --step features
    python main.py --step labels
    python main.py --step train
    python main.py --step backtest
"""

import argparse
import os
import sys
import time
import random
from pathlib import Path

import yaml
import numpy as np
from loguru import logger
from dotenv import load_dotenv

# Tải biến môi trường từ .env
load_dotenv()



# ---------------------------------------------------------------------------
# Config loader (merged từ config/__init__.py)
# ---------------------------------------------------------------------------

def load_config(config_path=None):
    """Load config từ YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths
    project_root = Path(__file__).parent
    paths = config.get("paths", {})
    for key, value in paths.items():
        if not os.path.isabs(value):
            paths[key] = str(project_root / value)
    config["paths"] = paths

    # Validate
    assert len(config["tickers"]) > 0, "Cần ít nhất 1 ticker"
    assert config["training"]["train_days"] > config["training"]["test_days"]

    return config


# ---------------------------------------------------------------------------
# Utility functions (merged từ src/utils.py)
# ---------------------------------------------------------------------------

def setup_logger(log_level="INFO", log_dir=None):
    """Setup loguru logger."""
    import io
    logger.remove()
    if hasattr(sys.stdout, 'buffer'):
        sink = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    else:
        # Colab/Jupyter notebook không có sys.stdout.buffer
        sink = sys.stdout

    logger.add(
        sink=sink, level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=False,
    )
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        logger.add(
            sink=os.path.join(log_dir, "pipeline_{time:YYYY-MM-DD}.log"),
            level="DEBUG", rotation="10 MB", retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        )
    return logger


def set_global_seed(seed=42):
    """Set random seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.info(f"Global random seed set to {seed}")


def ensure_dirs(config):
    """Tạo tất cả output directories."""
    for name, path in config.get("paths", {}).items():
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

STEPS = ["data", "features", "labels", "train", "backtest"]


def run_step(step_name, config):
    """Chạy một bước cụ thể của pipeline."""
    logger.info(f"{'='*60}")
    logger.info(f"STARTING STEP: {step_name.upper()}")
    logger.info(f"{'='*60}")
    start = time.time()

    if step_name == "data":
        from src.data_collection import run_data_collection
        run_data_collection(config)
    elif step_name == "features":
        from src.feature_engineering import run_feature_engineering
        run_feature_engineering(config)
    elif step_name == "labels":
        from src.label_engineering import run_label_engineering
        run_label_engineering(config)
    elif step_name == "train":
        # Chạy theo thứ tự CLAUDE.md: XGB/LGBM → LSTM → Ensemble
        from src.training.train_xgb_lgbm import run_xgb_lgbm_training
        run_xgb_lgbm_training(config)

        from src.training.train_lstm import run_lstm_training
        run_lstm_training(config)

        from src.training.train_ensemble import run_ensemble_training
        run_ensemble_training(config)

        # Export JSON contract cho LLM Agent
        from src.training.export_predictions import export_predictions
        export_predictions(config)
    elif step_name == "backtest":
        from src.backtesting import run_full_backtest
        run_full_backtest(config)
    else:
        logger.error(f"Unknown step: {step_name}")
        sys.exit(1)

    elapsed = time.time() - start
    logger.info(f"COMPLETED: {step_name.upper()} ({elapsed:.1f}s)")


def run_pipeline(config):
    """Chạy toàn bộ pipeline data chuẩn bị cho Colab."""
    total_start = time.time()
    logger.info("=" * 60)
    logger.info("VN30 STOCK PREDICTION PIPELINE — LOCAL DATA PREP")
    logger.info(f"Tickers: {len(config['tickers'])} mã")
    logger.info("=" * 60)

    # Kiến trúc Hybrid: Local chỉ chạy tới labels
    local_steps = ["data", "features", "labels"]
    for step in local_steps:
        run_step(step, config)

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"LOCAL DATA PREP COMPLETED — Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    logger.info("=" * 60)
    
    logger.success(
        "\n [HYBRID ARCHITECTURE] Dữ liệu đã chuẩn bị xong tại Local!\n\n"
        "BƯỚC TIẾP THEO:\n"
        "1. Nén hoặc copy thư mục `data/processed/` lên Google Drive của bạn.\n"
        "2. Mở file `notebooks/vn30_training_colab.ipynb` trên Google Colab.\n"
        "3. Chạy Notebook để huấn luyện Model bằng GPU và sinh ra JSON dự đoán.\n"
        "4. Tải thư mục `outputs/predictions/` từ Drive về lại máy Local.\n"
        "5. Chạy lệnh: `python main.py --mode chat` để hỏi đáp với Agent.\n"
    )


def run_chat(config):
    """Khởi chạy CLI chat interface."""
    from src.agent.cli import run_cli
    run_cli()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VN30 Stock Prediction Pipeline")
    parser.add_argument("--mode", choices=["pipeline", "chat", "train-tft"],
                        default=None, help="Chế độ chạy")
    parser.add_argument("--step", choices=STEPS, default=None,
                        help="Chạy một bước cụ thể")
    parser.add_argument("--base-path", default=None,
                        help="Base path cho persistent storage")
    parser.add_argument("--config", default=None,
                        help="Đường dẫn tới config file")

    args = parser.parse_args()

    config = load_config(args.config)
    setup_logger(
        log_level=config.get("log_level", "INFO"),
        log_dir=config["paths"].get("logs"),
    )
    set_global_seed(config.get("seed", 42))
    ensure_dirs(config)

    if args.mode == "chat":
        run_chat(config)
    elif args.mode == "train-tft":
        from src.training.train_tft import run_tft_training
        run_tft_training(config)
    elif args.mode == "pipeline":
        run_pipeline(config)
    elif args.step:
        run_step(args.step, config)
    else:
        run_pipeline(config)


if __name__ == "__main__":
    main()
