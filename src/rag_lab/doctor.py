from __future__ import annotations

import importlib.util
import socket
import subprocess


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> None:
    checks = {
        "chromadb": _has_module("chromadb"),
        "pymilvus": _has_module("pymilvus"),
        "milvus_lite": _has_module("milvus_lite"),
        "sklearn": _has_module("sklearn"),
        "rank_bm25": _has_module("rank_bm25"),
        "sentence_transformers_optional": _has_module("sentence_transformers"),
    }
    print("Python packages")
    for name, ok in checks.items():
        print(f"  {name:32} {'OK' if ok else 'missing'}")

    print("Services")
    print(f"  Milvus localhost:19530       {'open' if _port_open('localhost', 19530) else 'closed'}")
    try:
        output = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True)
        names = [line.strip() for line in output.splitlines() if line.strip()]
        print(f"  Docker reachable             OK ({len(names)} running containers)")
    except Exception as exc:
        print(f"  Docker reachable             failed ({exc})")


if __name__ == "__main__":
    main()
