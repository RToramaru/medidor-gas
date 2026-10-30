import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXECUTABLE = ROOT / "dist" / "LeitorHidrometros.exe"


def main():
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--noupx",
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
        "numpy",
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
    if not EXECUTABLE.is_file():
        raise FileNotFoundError(f"O executável não foi criado: {EXECUTABLE}")

    with tempfile.TemporaryDirectory(prefix="leitor-gas-smoke-") as temp_dir:
        log_path = Path(temp_dir) / "smoke-test.log"
        environment = os.environ.copy()
        environment["LEITOR_GAS_SMOKE_LOG"] = str(log_path)
        result = subprocess.run(
            [str(EXECUTABLE), "--smoke-test"],
            cwd=ROOT,
            env=environment,
            check=False,
            timeout=300,
        )
        if result.returncode:
            details = (
                log_path.read_text(encoding="utf-8")
                if log_path.is_file()
                else "O teste de inicialização terminou sem detalhes adicionais."
            )
            raise RuntimeError(
                f"O executável foi gerado, mas falhou no teste de inicialização "
                f"(código {result.returncode}):\n{details}"
            )

    print(f"Executável único validado: {EXECUTABLE}")


if __name__ == "__main__":
    main()
