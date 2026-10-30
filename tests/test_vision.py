import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from leitor_gas.vision import (
    _intersection_over_union,
    _prepare_tensor,
    _remove_overlaps,
    classify_roi,
    detect_digit_positions,
)


class TensorStub:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class BoxesStub:
    def __init__(self, coordinates, confidences, classes):
        self.xyxy = TensorStub(coordinates)
        self.conf = TensorStub(confidences)
        self.cls = TensorStub(classes)

    def __len__(self):
        return len(self.xyxy.value)


class DetectorResultStub:
    obb = None

    def __init__(self, boxes):
        self.boxes = boxes


class ClassifierStub:
    def __init__(self, scores):
        self.scores = np.asarray([scores], dtype=np.float32)
        self.tensor = None

    def set_tensor(self, index, tensor):
        self.tensor = tensor

    def invoke(self):
        return None

    def get_tensor(self, index):
        return self.scores


class VisionTests(unittest.TestCase):
    def test_preview_pipeline_does_not_draw_annotations(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "leitor_gas"
            / "application.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("cv2.rectangle", source)
        self.assertNotIn("cv2.putText", source)

    def test_tensor_preparation_resizes_and_returns_batch_dimension(self):
        model = {
            "width": 2,
            "height": 1,
            "input": {
                "dtype": np.dtype(np.uint8),
                "quantization": (0.0, 0),
            },
        }
        roi = np.zeros((2, 4, 3), dtype=np.uint8)

        tensor = _prepare_tensor(roi, model)

        self.assertEqual(tensor.shape, (1, 1, 2, 3))
        self.assertEqual(tensor.dtype, np.uint8)

    def test_classifier_returns_softmax_confidence_for_logits(self):
        interpreter = ClassifierStub([1.0, 3.0, 0.0])
        model = {
            "interpreter": interpreter,
            "input": {
                "index": 0,
                "dtype": np.dtype(np.uint8),
                "quantization": (0.0, 0),
            },
            "output": {
                "index": 0,
                "quantization": (0.0, 0),
            },
            "height": 2,
            "width": 2,
        }

        class_index, confidence = classify_roi(
            np.zeros((4, 4, 3), dtype=np.uint8), model
        )

        self.assertEqual(class_index, 1)
        self.assertAlmostEqual(confidence, 0.8438, places=3)
        self.assertEqual(interpreter.tensor.shape, (1, 2, 2, 3))

    def test_intersection_over_union_for_identical_boxes(self):
        box = {"x1": 1, "y1": 2, "x2": 11, "y2": 12}

        self.assertEqual(_intersection_over_union(box, box), 1.0)

    def test_overlap_filter_keeps_higher_confidence_and_sorts_left_to_right(self):
        boxes = [
            {"x1": 20, "y1": 0, "x2": 30, "y2": 10, "confidence": 0.8},
            {"x1": 22, "y1": 0, "x2": 32, "y2": 10, "confidence": 0.5},
            {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "confidence": 0.7},
        ]

        result = _remove_overlaps(boxes)

        self.assertEqual([box["x1"] for box in result], [0, 20])
        self.assertEqual(result[-1]["confidence"], 0.8)

    def test_digit_detection_filters_to_meter_window(self):
        boxes = BoxesStub(
            coordinates=[
                [10, 10, 90, 50],
                [20, 20, 30, 40],
                [22, 20, 32, 40],
                [95, 20, 105, 40],
            ],
            confidences=[0.99, 0.7, 0.9, 0.8],
            classes=[1, 2, 3, 4],
        )
        result = DetectorResultStub(boxes)
        model = type(
            "ModelStub",
            (),
            {"names": {0: "meter", 1: "window", 2: "0", 3: "1", 4: "2"}},
        )()
        image = np.zeros((100, 120, 3), dtype=np.uint8)

        with patch(
            "leitor_gas.vision._run_detector",
            return_value=(result, model, "CPU"),
        ):
            digits, backend = detect_digit_positions(image)

        self.assertEqual(backend, "CPU")
        self.assertEqual(len(digits), 1)
        self.assertEqual(digits[0]["name"], "1")
        self.assertEqual(digits[0]["x1"], 22.0)


if __name__ == "__main__":
    unittest.main()
