import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def _run_smoke_test():
    try:
        import numpy as np

        import leitor_gas.application
        from leitor_gas.settings import DECIMAL_CLASSIFIER, INTEGER_CLASSIFIER
        from leitor_gas.vision import classify_roi, load_classifier

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
