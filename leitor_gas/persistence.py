import csv
import json
import shutil
from pathlib import Path

from leitor_gas.settings import CALIBRATIONS_DIR, LEGACY_RESULTS_DIR, RESULTS_DIR


def load_calibration(path: Path) -> dict | None:
    if not path.is_file():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or int(data["width"]) <= 0
        or int(data["height"]) <= 0
        or not isinstance(data["rois"], list)
        or not data["rois"]
    ):
        raise ValueError("O arquivo de calibração está incompleto.")

    width = int(data["width"])
    height = int(data["height"])
    for roi in data["rois"]:
        x1, y1, x2, y2 = (int(roi[key]) for key in ("x1", "y1", "x2", "y2"))
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("O arquivo de calibração contém regiões inválidas.")

    return data


def save_calibration(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)


def update_calibration_regions(data: dict, coordinates: list[tuple[int, int, int, int]]) -> dict:
    width = int(data["width"])
    height = int(data["height"])
    if len(coordinates) != len(data["rois"]):
        raise ValueError("A quantidade de regiões não corresponde à calibração.")

    rois = []
    for index, (roi, bounds) in enumerate(zip(data["rois"], coordinates), start=1):
        x1, y1, x2, y2 = bounds
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Os limites do dígito {index} são inválidos.")
        updated_roi = dict(roi)
        updated_roi.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        rois.append(updated_roi)

    result = dict(data)
    result["rois"] = sorted(rois, key=lambda roi: (roi["x1"] + roi["x2"]) / 2)
    return result


def append_reading(path: Path, row: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        if is_new_file:
            writer.writerow(["data_hora", "fonte", "leitura", "backend", "rois"])
        writer.writerow(row)


def migrate_legacy_results() -> str | None:
    legacy_calibrations = LEGACY_RESULTS_DIR / "calibracoes"
    try:
        if legacy_calibrations.is_dir():
            CALIBRATIONS_DIR.mkdir(parents=True, exist_ok=True)
            for source in legacy_calibrations.glob("*.json"):
                destination = CALIBRATIONS_DIR / source.name
                if not destination.exists():
                    shutil.copy2(source, destination)

        legacy_csv = LEGACY_RESULTS_DIR / "leituras.csv"
        destination_csv = RESULTS_DIR / "leituras.csv"
        if legacy_csv.is_file() and not destination_csv.exists():
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_csv, destination_csv)
    except OSError as exc:
        return f"Não foi possível migrar dados antigos: {exc}"
    return None
