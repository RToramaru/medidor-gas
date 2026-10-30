from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from leitor_gas.settings import (
    DETECTION_CONFIDENCE,
    DETECTION_IMAGE_SIZE,
    DETECTOR_MODEL,
    MAXIMUM_IOU,
    OPENVINO_MODEL,
)


def load_classifier(path: Path, expected_classes: int) -> dict:
    from ai_edge_litert.interpreter import Interpreter

    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {path}")

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_shape = input_details["shape"].tolist()

    if len(input_shape) != 4 or int(input_shape[-1]) != 3:
        raise ValueError(f"Formato de entrada inesperado em {path.name}: {input_shape}")

    output_classes = int(np.prod(output_details["shape"][1:]))
    if output_classes != expected_classes:
        raise ValueError(
            f"{path.name}: saída com {output_classes} classes; "
            f"esperadas {expected_classes}"
        )

    return {
        "interpreter": interpreter,
        "input": input_details,
        "output": output_details,
        "height": int(input_shape[1]),
        "width": int(input_shape[2]),
    }


def _prepare_tensor(roi: np.ndarray, model: dict) -> np.ndarray:
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (model["width"], model["height"]), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(image, dtype=np.float32)
    input_details = model["input"]
    input_dtype = input_details["dtype"]

    if input_dtype == np.float32:
        tensor = pixels
    elif np.issubdtype(input_dtype, np.integer):
        scale, zero_point = input_details["quantization"]
        values = pixels / 255.0 if 0 < scale < 0.01 else pixels
        if scale > 0:
            values = np.round(values / scale + zero_point)
        limits = np.iinfo(input_dtype)
        tensor = np.clip(values, limits.min, limits.max).astype(input_dtype)
    else:
        raise TypeError(f"Tipo de entrada TFLite não suportado: {input_dtype}")

    return np.expand_dims(tensor, axis=0)


def classify_roi(roi: np.ndarray, model: dict) -> tuple[int, float]:
    interpreter = model["interpreter"]
    input_details = model["input"]
    output_details = model["output"]

    interpreter.set_tensor(input_details["index"], _prepare_tensor(roi, model))
    interpreter.invoke()
    scores = np.asarray(interpreter.get_tensor(output_details["index"])[0]).flatten()

    if np.issubdtype(scores.dtype, np.integer):
        scale, zero_point = output_details["quantization"]
        if scale > 0:
            scores = (scores.astype(np.float32) - zero_point) * scale
        else:
            scores = scores.astype(np.float32)

    scores = scores.astype(np.float32)
    class_index = int(np.argmax(scores))
    if np.all(scores >= 0) and np.all(scores <= 1) and np.isclose(
        float(scores.sum()), 1.0, atol=0.05
    ):
        confidence = float(scores[class_index])
    else:
        shifted = scores - scores.max()
        probabilities = np.exp(shifted)
        confidence = float(probabilities[class_index] / probabilities.sum())
    return class_index, confidence


def _predict(image: np.ndarray, model_path: Path, device: str | int) -> tuple:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.predict(
        source=image,
        conf=DETECTION_CONFIDENCE,
        imgsz=DETECTION_IMAGE_SIZE,
        device=device,
        verbose=False,
    )
    if not results:
        raise RuntimeError("O detector não retornou resultados.")

    result = results[0]
    oriented_boxes = getattr(result, "obb", None)
    boxes = getattr(result, "boxes", None)
    count = (
        len(oriented_boxes)
        if oriented_boxes is not None
        else len(boxes) if boxes is not None else 0
    )
    if count == 0:
        raise RuntimeError("Nenhum dígito foi localizado na imagem.")

    return result, model


def _run_detector(image: np.ndarray) -> tuple:
    errors = []

    try:
        import torch

        if torch.cuda.is_available():
            try:
                result, model = _predict(image, DETECTOR_MODEL, 0)
                return result, model, "CUDA"
            except Exception as exc:
                errors.append(f"GPU NVIDIA: {exc}")
    except Exception as exc:
        errors.append(f"GPU NVIDIA indisponível: {exc}")

    if OPENVINO_MODEL.is_dir():
        try:
            import openvino as ov

            devices = ov.Core().available_devices
            if any(str(device).upper().startswith("GPU") for device in devices):
                try:
                    result, model = _predict(image, OPENVINO_MODEL, "intel:gpu")
                    return result, model, "GPU Intel"
                except Exception as exc:
                    errors.append(f"GPU Intel: {exc}")
        except Exception as exc:
            errors.append(f"OpenVINO indisponível: {exc}")

        try:
            result, model = _predict(image, OPENVINO_MODEL, "intel:cpu")
            return result, model, "CPU"
        except Exception as exc:
            errors.append(f"CPU: {exc}")

    try:
        result, model = _predict(image, DETECTOR_MODEL, "cpu")
        return result, model, "CPU"
    except Exception as exc:
        errors.append(f"CPU: {exc}")

    raise RuntimeError(
        "Não foi possível localizar os dígitos.\n\n" + "\n".join(errors)
    )


def _area(box: dict) -> float:
    return max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"])


def _intersection_over_union(first: dict, second: dict) -> float:
    x1 = max(first["x1"], second["x1"])
    y1 = max(first["y1"], second["y1"])
    x2 = min(first["x2"], second["x2"])
    y2 = min(first["y2"], second["y2"])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union else 0.0


def _boxes_from_result(result) -> list[dict]:
    oriented_boxes = getattr(result, "obb", None)
    if oriented_boxes is not None and len(oriented_boxes):
        detections = zip(
            oriented_boxes.xyxy.cpu().tolist(),
            oriented_boxes.conf.cpu().tolist(),
            oriented_boxes.cls.cpu().tolist(),
        )
    else:
        boxes = getattr(result, "boxes", None)
        if boxes is None or not len(boxes):
            return []
        detections = zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
        )

    return [
        {
            "x1": float(coords[0]),
            "y1": float(coords[1]),
            "x2": float(coords[2]),
            "y2": float(coords[3]),
            "confidence": float(confidence),
            "class_id": int(class_id),
        }
        for coords, confidence, class_id in detections
    ]


def _remove_overlaps(boxes: list[dict]) -> list[dict]:
    selected = []
    for box in sorted(boxes, key=lambda item: item["confidence"], reverse=True):
        if not any(
            _intersection_over_union(box, other) > MAXIMUM_IOU for other in selected
        ):
            selected.append(box)

    return sorted(selected, key=lambda box: (box["x1"] + box["x2"]) / 2)


def detect_digit_positions(image: np.ndarray) -> tuple[list[dict], str]:
    result, model, backend = _run_detector(image)
    digits = []
    windows = []

    for box in _boxes_from_result(result):
        name = str(model.names[box["class_id"]]).strip().lower()
        box["name"] = name
        if "window" in name:
            windows.append(box)
        elif name in {str(number) for number in range(10)} or name == "u":
            digits.append(box)

    if windows:
        window = max(windows, key=_area)
        digits = [
            box
            for box in digits
            if box["x2"] > window["x1"]
            and box["x1"] < window["x2"]
            and box["y2"] > window["y1"]
            and box["y1"] < window["y2"]
        ]

    digits = _remove_overlaps(digits)
    if not digits:
        raise RuntimeError(
            "Nenhum dígito foi localizado. Verifique a imagem e tente calibrar novamente."
        )

    return digits, backend
