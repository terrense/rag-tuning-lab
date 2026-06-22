"""
================================================================================
doctor.py —— 环境体检
--------------------------------------------------------------------------------
用法：python -m rag_lab.doctor
检查依赖包是否装好、Milvus 端口通不通、Docker 是否可用。排环境问题时先跑它。
================================================================================
"""

from __future__ import annotations

import importlib.util
import socket
import subprocess


def _has_module(name: str) -> bool:
    """这个包能不能 import（不真的 import，只查有没有）。"""
    return importlib.util.find_spec(name) is not None


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """探测某个 host:port 能不能连上（用来判断 Milvus 服务是否在跑）。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> None:
    # 1) 关键 Python 包是否就位
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

    # 2) 服务：Milvus 端口 + Docker
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
