import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from leitor_gas.application import main


if __name__ == "__main__":
    main()
