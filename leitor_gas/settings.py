import os
from pathlib import Path
import sys


PROJECT_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent.parent
)
ROOT_DIR = Path(getattr(sys, "_MEIPASS", PROJECT_DIR))
MODELS_DIR = ROOT_DIR / "modelos"
APP_ICON_PATH = ROOT_DIR / "imagens" / "icone.ico"

if sys.platform == "win32":
    APP_DATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "LeitorGas"
elif sys.platform == "darwin":
    APP_DATA_DIR = Path.home() / "Library" / "Application Support" / "LeitorGas"
else:
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    APP_DATA_DIR = Path(
        xdg_data_home or Path.home() / ".local" / "share"
    ) / "leitor-gas"

RESULTS_DIR = APP_DATA_DIR / "resultados"
CALIBRATIONS_DIR = RESULTS_DIR / "calibracoes"
LEGACY_RESULTS_DIR = PROJECT_DIR / "resultados"

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
