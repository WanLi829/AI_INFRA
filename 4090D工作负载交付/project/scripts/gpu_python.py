"""Use the private Python/CUDA installation and keep generated files in its workspace."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]


def cuda_toolkit():
    distribution = importlib.metadata.distribution("nvidia-cuda-nvcc")
    for entry in distribution.files or ():
        if str(entry).endswith("/bin/nvcc"):
            return Path(distribution.locate_file(entry)).resolve().parent.parent
    raise RuntimeError("The private CUDA compiler is not installed")


def prepare_environment(base, cuda_home, inherited):
    base, cuda_home = Path(base).resolve(), Path(cuda_home).resolve()
    if not cuda_home.is_relative_to(base / "env"):
        raise ValueError("CUDA must belong to this workspace's isolated environment")
    for path in (cuda_home / "bin/nvcc", cuda_home / "include/cuda_runtime.h"):
        if not path.is_file():
            raise RuntimeError("Incomplete private CUDA installation: " + str(path))
    paths = dict(TMPDIR=base / "tmp", XDG_CACHE_HOME=base / "cache", XDG_CONFIG_HOME=base / "config",
        HF_HOME=base / "cache/huggingface", TORCH_HOME=base / "cache/torch",
        TORCH_EXTENSIONS_DIR=base / "cache/extensions", TRITON_CACHE_DIR=base / "cache/triton",
        CUDA_CACHE_PATH=base / "cache/cuda", TORCHINDUCTOR_CACHE_DIR=base / "cache/inductor",
        FLASHINFER_WORKSPACE_BASE=base / "cache/flashinfer", CUPY_CACHE_DIR=base / "cache/cupy",
        NUMBA_CACHE_DIR=base / "cache/numba", MPLCONFIGDIR=base / "config/matplotlib",
        HF_DATASETS_CACHE=base / "cache/hf_datasets", RAY_TMPDIR=base / "tmp/ray")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    selected = {name: str(path) for name, path in paths.items()}
    selected.update(CUDA_HOME=str(cuda_home), CUDACXX=str(cuda_home / "bin/nvcc"),
                    MALLOC_ARENA_MAX="2", PYTHONDONTWRITEBYTECODE="1",
                    PYTHONPATH=str(base / "project"), PYTHONNOUSERSITE="1")
    # Container driver mounts can work for the loader without appearing in ldconfig.
    # Give Triton the actual driver directory rather than a CUDA toolkit stub.
    candidates = [inherited.get("TRITON_LIBCUDA_PATH"), "/usr/lib/x86_64-linux-gnu",
                  "/usr/local/nvidia/lib64", "/usr/lib64"]
    for candidate in candidates:
        if candidate and "stubs" not in Path(candidate).parts and (Path(candidate) / "libcuda.so.1").is_file():
            selected["TRITON_LIBCUDA_PATH"] = str(Path(candidate).resolve())
            break
    env = dict(inherited)
    env.update(selected)
    env["PATH"] = ":".join((str(base / "env/bin"), str(cuda_home / "bin"), env.get("PATH", "")))
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, (str(cuda_home / "lib"), env.get("LD_LIBRARY_PATH", ""))))
    return env, selected


def main():
    base = PROJECT.parent
    if Path(sys.prefix).resolve() != (base / "env").resolve():
        raise RuntimeError("Invoke this script with " + str(base / "env/bin/python"))
    env, selected = prepare_environment(base, cuda_toolkit(), os.environ)
    if sys.argv[1:] == ["--env-plan"]:
        version = subprocess.check_output([selected["CUDACXX"], "--version"], text=True)
        print(json.dumps(dict(environment=selected, nvcc=version, gpu_started=False), indent=2))
        return
    if len(sys.argv) == 1:
        raise SystemExit("Usage: ../env/bin/python scripts/gpu_python.py -m b_replay server-run ...")
    os.chdir(PROJECT)
    os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
