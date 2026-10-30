import unittest
from unittest.mock import patch

import numpy as np

from leitor_gas.vision import (
    _intersection_over_union,
    _remove_overlaps,
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


class VisionTests(unittest.TestCase):
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
