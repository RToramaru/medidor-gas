import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        "LeitorHidrometros",
        "--icon",
        str(ROOT / "imagens" / "icone.ico"),
        "--paths",
        str(ROOT),
        "--add-data",
        f"{ROOT / 'modelos'}{os.pathsep}modelos",
        "--add-data",
        f"{ROOT / 'imagens' / 'icone.ico'}{os.pathsep}imagens",
        "--collect-all",
        "ai_edge_litert",
        "--collect-all",
        "customtkinter",
        "--hidden-import",
        "torch",
        "--collect-data",
        "ultralytics",
        str(ROOT / "main.py"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Aplicativo gerado em: {ROOT / 'dist' / 'LeitorHidrometros'}")


if __name__ == "__main__":
    main()
