from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def install_ollama_if_requested(install_ollama: bool) -> None:
    if not install_ollama:
        return
    run_cmd(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"])


def start_ollama_server(base_url: str) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    # Default Ollama listen address in Colab.
    env.setdefault("OLLAMA_HOST", "127.0.0.1:11434")

    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    wait_until_ollama_ready(base_url=base_url, timeout_seconds=120)
    return proc


def wait_until_ollama_ready(base_url: str, timeout_seconds: int = 120) -> None:
    deadline = time.time() + timeout_seconds
    url = f"{base_url.rstrip('/')}/api/tags"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    print("Ollama server is ready.")
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError(f"Ollama server did not become ready within {timeout_seconds}s")


def ollama_pull(model: str) -> None:
    run_cmd(["ollama", "pull", model])


def ollama_warmup(model: str, base_url: str, keep_alive: str, num_ctx: int) -> None:
    payload = {
        "model": model,
        "prompt": '{"질병명":[],"증상":[],"처치 방법":[]}',
        "stream": False,
        "keep_alive": keep_alive,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_ctx": num_ctx,
        },
    }
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as resp:
        _ = resp.read().decode("utf-8")
    print(f"Model warmed up: {model} (keep_alive={keep_alive})")


def run_preprocess(
    repo_dir: Path,
    model: str,
    base_url: str,
    keep_alive: str,
    timeout_seconds: int,
    num_ctx: int,
    max_records: int | None,
) -> None:
    cmd = [
        sys.executable,
        "src/preprocess_for_chroma.py",
        "--model",
        model,
        "--ollama-url",
        base_url,
        "--keep-alive",
        keep_alive,
        "--timeout-seconds",
        str(timeout_seconds),
        "--num-ctx",
        str(num_ctx),
    ]
    if max_records is not None:
        cmd.extend(["--max-records", str(max_records)])

    run_cmd(cmd, cwd=repo_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colab에서 Ollama + 전처리 파이프라인을 한 번에 실행합니다."
    )
    parser.add_argument(
        "--repo-dir",
        default="/content/dermatology-rag-chatbot",
        help="프로젝트 루트 경로",
    )
    parser.add_argument(
        "--install-ollama",
        action="store_true",
        help="Ollama를 설치합니다(Colab 첫 실행 시 사용).",
    )
    parser.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama 모델명")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--keep-alive", default="30m", help="모델 keep_alive")
    parser.add_argument("--timeout-seconds", type=int, default=600, help="요청 타임아웃")
    parser.add_argument("--num-ctx", type=int, default=4096, help="컨텍스트 길이")
    parser.add_argument("--max-records", type=int, default=None, help="샘플 처리 개수 제한")
    parser.add_argument("--skip-pull", action="store_true", help="모델 pull 단계를 건너뜁니다.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    if not repo_dir.exists():
        raise FileNotFoundError(f"repo dir not found: {repo_dir}")

    install_ollama_if_requested(args.install_ollama)
    _server_proc = start_ollama_server(base_url=args.ollama_url)

    if not args.skip_pull:
        ollama_pull(args.model)
    ollama_warmup(
        model=args.model,
        base_url=args.ollama_url,
        keep_alive=args.keep_alive,
        num_ctx=args.num_ctx,
    )

    run_preprocess(
        repo_dir=repo_dir,
        model=args.model,
        base_url=args.ollama_url,
        keep_alive=args.keep_alive,
        timeout_seconds=args.timeout_seconds,
        num_ctx=args.num_ctx,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()

