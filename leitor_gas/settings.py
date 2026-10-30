from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "modelos"
RESULTS_DIR = ROOT_DIR / "resultados"
CALIBRATIONS_DIR = RESULTS_DIR / "calibracoes"

DETECTOR_MODEL = MODELS_DIR / "stage1_meter_window_digits.pt"
OPENVINO_MODEL = MODELS_DIR / "stage1_meter_window_digits_openvino_model"
INTEGER_CLASSIFIER = MODELS_DIR / "dig-class11_1910_s2_q.tflite"
DECIMAL_CLASSIFIER = MODELS_DIR / "dig-class100-0182-s2_q.tflite"

DETECTION_CONFIDENCE = 0.15
DETECTION_IMAGE_SIZE = 640
MAXIMUM_IOU = 0.30
FRAME_INTERVAL_MS = 35

BACKGROUND = "#111318"
PANEL = "#1A1D24"
BORDER = "#303641"
GREEN = "#36C275"
RED = "#E05252"
TEXT = "#F2F4F8"
MUTED = "#9AA4B2"
