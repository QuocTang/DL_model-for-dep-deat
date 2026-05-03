"""Cấu hình đường dẫn chung cho v2.

Mọi data đầu vào, model weights, features, kết quả ML đều nằm dưới `v2/`.
Các script con import từ đây thay vì hard-code đường dẫn.
"""

from pathlib import Path

V2_ROOT = Path(__file__).resolve().parent

# === Datasets ===
SRC_ROOT = V2_ROOT / "data" / "durian-1"          # Dataset gốc (sau khi pull từ Roboflow)
DST_ROOT = V2_ROOT / "data" / "durian_refactor"   # Dataset sau khi refactor nhãn
DATA_YAML = DST_ROOT / "data.yaml"                # YAML cho dataset đã refactor (6 class)

# === Outputs (gộp chung dưới `v2/output/`) ===
OUTPUT_ROOT = V2_ROOT / "output"

# YOLO training
RUNS_DIR = OUTPUT_ROOT / "runs" / "detect"
DEFAULT_RUN_NAME = "durian-1-yolo11x"
DEFAULT_WEIGHTS = RUNS_DIR / DEFAULT_RUN_NAME / "weights" / "best.pt"

# EDA
EDA_DIR = OUTPUT_ROOT / "eda"

# Feature extraction & ML training
EXTRACTED_FEATURES_DIR = OUTPUT_ROOT / "extracted_features"
EXTRACTED_FEATURES_TOP6 = OUTPUT_ROOT / "extracted_features_top6"
ML_MODELS_DIR = OUTPUT_ROOT / "ml_models"
MODEL_COMPARISON_DIR = OUTPUT_ROOT / "model_comparison"

# === Inference defaults (XGBoost top-6) ===
DEFAULT_ML_MODEL = MODEL_COMPARISON_DIR / "top6_filtered" / "xgb" / "xgb_model.joblib"
DEFAULT_RESULTS_JSON = MODEL_COMPARISON_DIR / "top6_filtered" / "xgb" / "results.json"
