from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request


OLLAMA_HOST = "http://127.0.0.1:11434"


def run_command(args: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return 127, ""
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def check_ollama_api() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = [model.get("name", "") for model in data.get("models", [])]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        print("Ollama API: not reachable at http://127.0.0.1:11434")
        print("Start Ollama first, then rerun this script.")
        return False

    print("Ollama API: reachable")
    print("Installed models:")
    for name in models:
        print(f"  - {name}")
    return True


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")

    ollama_path = shutil.which("ollama")
    if ollama_path:
        print(f"Ollama executable: {ollama_path}")
    else:
        print("Ollama executable: not found in PATH")
        return 1

    code, output = run_command(["ollama", "list"])
    if code == 0:
        print("\nollama list:")
        print(output)
    else:
        print("Could not run 'ollama list'.")
        print(output)
        return 1

    nvidia_path = shutil.which("nvidia-smi")
    if nvidia_path:
        print(f"\nnvidia-smi executable: {nvidia_path}")
        code, output = run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        if code == 0:
            print("GPU memory:")
            print(output)
        else:
            print("nvidia-smi exists but memory query failed:")
            print(output)
    else:
        print("\nnvidia-smi executable: not found. VRAM metrics will be unavailable.")

    print()
    return 0 if check_ollama_api() else 1


if __name__ == "__main__":
    raise SystemExit(main())

