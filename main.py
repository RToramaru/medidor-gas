import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def _run_smoke_test():
    try:
        import numpy as np
        import openvino as ov
        import torch
        from torchvision.ops import nms

        import leitor_gas.application
        from leitor_gas.settings import (
            DECIMAL_CLASSIFIER,
            INTEGER_CLASSIFIER,
            OPENVINO_MODEL,
        )
        from leitor_gas.vision import classify_roi, load_classifier

        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8])
        if nms(boxes, scores, 0.5).tolist() != [0]:
            raise RuntimeError("A extensão torchvision NMS não está funcional.")

        core = ov.Core()
        if "CPU" not in core.available_devices:
            raise RuntimeError("O plugin OpenVINO para CPU não está disponível.")
        xml_model = next(OPENVINO_MODEL.glob("*.xml"))
        compiled_model = core.compile_model(str(xml_model), "CPU")
        input_layer = compiled_model.input(0)
        input_shape = tuple(int(size) for size in input_layer.shape)
        input_data = np.zeros(
            input_shape, dtype=input_layer.get_element_type().to_dtype()
        )
        compiled_model({input_layer: input_data})

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        integer_model = load_classifier(INTEGER_CLASSIFIER, 11)
        decimal_model = load_classifier(DECIMAL_CLASSIFIER, 100)
        classify_roi(frame, integer_model)
        classify_roi(frame, decimal_model)
    except Exception:
        log_path = Path(
            os.environ.get(
                "LEITOR_GAS_SMOKE_LOG",
                Path(tempfile.gettempdir()) / "leitor-gas-smoke-test.log",
            )
        )
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise SystemExit(1)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _run_smoke_test()
        raise SystemExit(0)

    from leitor_gas.application import main

    main()
