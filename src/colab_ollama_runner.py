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


def install_python_deps_if_requested(install_python_deps: bool) -> None:
    if not install_python_deps:
        return
    # preprocess_for_chroma.py uses tqdm.
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "tqdm"])


def normalize_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/api"):
        url = url[:-4]
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def start_ollama_server(base_url: str) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
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
    url = f"{normalize_base_url(base_url)}/api/tags"
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
        "prompt": '{"질병명":[],"증상":[],"처치 방법":[],"환자 특성":[]}',
        "stream": False,
        "keep_alive": keep_alive,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_ctx": num_ctx,
        },
    }
    request = urllib.request.Request(
        url=f"{normalize_base_url(base_url)}/api/generate",
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
    inputs: list[str] | None,
    checkpoint_every: int,
) -> None:
    cmd = [
        sys.executable,
        "src/preprocess_for_chroma.py",
        "--model",
        model,
        "--ollama-url",
        normalize_base_url(base_url),
        "--keep-alive",
        keep_alive,
        "--timeout-seconds",
        str(timeout_seconds),
        "--num-ctx",
        str(num_ctx),
    ]
    if max_records is not None:
        cmd.extend(["--max-records", str(max_records)])
    cmd.extend(["--checkpoint-every", str(checkpoint_every)])
    if inputs:
        cmd.extend(["--inputs", *inputs])

    run_cmd(cmd, cwd=repo_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colab에서 Ollama + preprocess_for_chroma.py를 실행합니다."
    )
    parser.add_argument(
        "--repo-dir",
        default="/content/dermatology-rag-chatbot",
        help="프로젝트 루트 경로",
    )
    parser.add_argument(
        "--install-ollama",
        action="store_true",
        help="Colab에서 Ollama를 설치합니다.",
    )
    parser.add_argument(
        "--install-python-deps",
        action="store_true",
        help="Python 의존성(tqdm)을 설치합니다.",
    )
    parser.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama 모델명")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--keep-alive", default="90m", help="모델 keep_alive (예: 30m, 90m, 2h)")
    parser.add_argument("--timeout-seconds", type=int, default=600, help="요청 타임아웃(초)")
    parser.add_argument("--num-ctx", type=int, default=8192, help="컨텍스트 길이")
    parser.add_argument("--max-records", type=int, default=None, help="샘플 처리 개수 제한")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="중간 저장 주기")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["data/raw/TL_내과_통합.json"],
        help="preprocess 입력 JSON 파일 목록",
    )
    parser.add_argument("--skip-pull", action="store_true", help="모델 pull 단계를 건너뜁니다.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    if not repo_dir.exists():
        raise FileNotFoundError(f"repo dir not found: {repo_dir}")

    install_ollama_if_requested(args.install_ollama)
    install_python_deps_if_requested(args.install_python_deps)

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
        inputs=args.inputs,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
