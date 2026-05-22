# 🏠 House Price Prediction API

พยากรณ์ราคาบ้านด้วย XGBoost / LightGBM / Ensemble Model

## 📋 Requirements
- Python 3.11+
- Docker

## 🚀 Installation

### Run with Docker
```bash
docker compose up --build
```

### Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/options` | ดูค่า zone / house_type |
| POST | `/predict` | พยากรณ์ทั้ง 3 models |
| POST | `/predict/xgboost` | XGBoost เท่านั้น |
| POST | `/predict/lightgbm` | LightGBM เท่านั้น |
| POST | `/predict/debug` | Debug mode |

## 📊 Model Performance

| Model | R² | RMSE | MAPE |
|-------|----|------|------|
| XGBoost | 0.9687 | 1,558,062 ฿ | 12.78% |
| LightGBM | 0.9674 | 1,588,343 ฿ | 14.87% |
| Ensemble | 0.9705 | 1,512,503 ฿ | 13.09% |

## 📁 Project Structure
```
├── app/
│   └── main.py
├── models/
│   ├── xgb_model.joblib
│   ├── lgb_model.joblib
│   ├── ordinal_encoder.joblib
│   └── feature_config.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```