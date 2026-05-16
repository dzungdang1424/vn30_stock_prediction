# VN30 Stock Prediction — Hybrid AI System

Hệ thống dự đoán xu hướng giá cổ phiếu T+1 và T+3 cho **toàn bộ 30 mã VN30**, kết hợp Machine Learning + Technical Analysis + LLM Agent.

> ⚠️ **Disclaimer:** Đây là dự án nghiên cứu học thuật. Kết quả từ mô hình AI **không phải khuyến nghị đầu tư**.

---

## Kiến trúc tổng quan

Hệ thống sử dụng **Kiến trúc Thực thi Lai (Hybrid Execution)** — tách biệt hoàn toàn ML Pipeline và LLM Agent, giao tiếp qua file JSON:

```
[Local] Data Collection → Feature Engineering → Label Engineering
                                    ↓
                     (Tải thủ công lên Google Drive)
                                    ↓
           [Google Colab GPU] Training (XGB, LGBM, LSTM, Ensemble)
                                    ↓
                     (Tải thủ công về Local — outputs/predictions/)
                                    ↓
                    [Local] LLM Agent đọc JSON → Gemini API → CLI
```

---

## Cấu trúc dự án

```
vn30_stock_prediction/
├── .env.example                  # Mẫu cấu hình API keys
├── config.yaml                   # Tickers, paths, hyperparameters
├── main.py                       # Entry point
├── Makefile
├── requirements.txt
├── notebooks/
│   ├── vn30_training_colab.ipynb # Training pipeline (chạy trên Colab)
│   └── vn30_chat_colab.ipynb     # Chat/demo notebook
├── src/
│   ├── data_collection.py        # Fetch OHLCV từ vnstock
│   ├── feature_engineering.py    # Technical indicators
│   ├── label_engineering.py      # Future returns T+1, T+3
│   ├── backtesting.py            # Tính Sharpe ratio
│   ├── training/
│   │   ├── train_xgb_lgbm.py     # XGBoost + LightGBM
│   │   ├── train_lstm.py         # LSTM
│   │   ├── train_ensemble.py     # Stacking Ensemble
│   │   ├── train_tft.py          # Temporal Fusion Transformer
│   │   ├── export_predictions.py # Export JSON contract
│   │   └── _training_utils.py    # Utilities dùng chung
│   └── agent/
│       ├── cli.py                # Vòng lặp hỏi-đáp
│       ├── llm_agent.py          # Build prompt, gọi Gemini
│       ├── llm_client.py         # Gemini REST API wrapper
│       ├── ticker_resolver.py    # Fuzzy match mã cổ phiếu
│       ├── ticker_map.json       # Mapping tên → mã
│       └── prompt_templates.py   # Prompt templates
├── data/
│   └── processed/
│       ├── ohlcv/                # Dữ liệu giá thô (.parquet)
│       ├── features/             # Technical indicators (.parquet)
│       └── labels/               # Future returns (.parquet)
└── outputs/
    ├── metrics/                  # Kết quả train (JSON, PKL)
    ├── predictions/              # JSON contract ML ↔ LLM Agent
    └── logs/                     # Pipeline logs
```

---

## Yêu cầu

- Python 3.10+
- Tài khoản Google (để dùng Colab + Gemini API)
- API key Vnstock (xem hướng dẫn bên dưới)

---

## Cài đặt

### 1. Clone repo

```bash
git clone https://github.com/<your-username>/vn30_stock_prediction.git
cd vn30_stock_prediction
```

### 2. Tạo virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Cài dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình API keys

```bash
cp .env.example .env
```

Mở file `.env` và điền key của bạn:

```env
GEMINI_API_KEY=...   # https://aistudio.google.com/app/apikey (miễn phí)
VNSTOCK_API_KEY=...  # https://docs.vnstock.site
```

---

## Hướng dẫn sử dụng

### Bước 1 — Thu thập dữ liệu (Local)

```bash
python main.py --mode data
```

Kết quả lưu tại `data/processed/ohlcv/`.

### Bước 2 — Feature & Label Engineering (Local)

```bash
python main.py --mode features
python main.py --mode labels
```

### Bước 3 — Training (Google Colab)

1. Upload thư mục `data/processed/` lên Google Drive.
2. Mở `notebooks/vn30_training_colab.ipynb` trên Google Colab.
3. Chạy toàn bộ notebook (GPU runtime).
4. Tải thư mục `outputs/predictions/` về Local.

### Bước 4 — Chạy LLM Agent (Local)

```bash
python main.py --mode chat
```

Ví dụ hỏi:
```
> FPT ngày mai thế nào?
> Phân tích VCB và HPG
> VN30 hôm nay có mã nào đáng chú ý không?
```

---

## Mô hình sử dụng

| Model | Mục đích |
|-------|----------|
| XGBoost + LightGBM | Base learners, Walk-Forward Validation |
| LSTM | Sequence modeling (20 ngày) |
| Stacking Ensemble | Meta-learner (Logistic Regression) |
| TFT | Temporal Fusion Transformer (experimental) |
| Gemini 2.5 Flash | LLM giải thích kết quả dự đoán |

---

## Nguồn dữ liệu

- **Giá cổ phiếu:** [vnstock](https://docs.vnstock.site) — nguồn `vci` hoặc `kbs`
- **Danh sách VN30:** Hardcode trong `config.yaml`, review 2 lần/năm (tháng 1 và tháng 7)

---

## Lưu ý quan trọng

- File `.env` chứa API keys — **không bao giờ commit lên Git**.
- Thư mục `data/` và `outputs/metrics/` không được push lên GitHub (dữ liệu lớn).
- Free tier Gemini AI Studio: ~15 RPM, ~1,500 RPD — đủ dùng cho CLI hỏi-đáp.
- Dự án chấp nhận **Survivorship Bias** từ danh sách VN30 hiện tại (xem giới hạn đã biết).
