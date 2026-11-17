import importlib.util

import pytest

from b_replay.common import ROOT

spec = importlib.util.spec_from_file_location("gpu_python", ROOT / "scripts/gpu_python.py")
gpu_python = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_python)


def test_private_cuda_and_caches_without_home_override_or_secret_export(tmp_path):
    cuda = tmp_path / "env/lib/python3.12/site-packages/nvidia/cu13"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "include").mkdir()
    (cuda / "bin/nvcc").write_text("compiler placeholder; not executed")
    (cuda / "include/cuda_runtime.h").write_text("header placeholder")
    env, recorded = gpu_python.prepare_environment(tmp_path, cuda,
        {"HOME": "/root", "SECRET_PROBE": "must-not-be-recorded", "PATH": "/usr/bin"})
    assert env["HOME"] == "/root" and "HOME" not in recorded and "SECRET_PROBE" not in recorded
    assert env["CUDA_HOME"] == str(cuda) and env["CUDACXX"] == str(cuda / "bin/nvcc")
    assert all((tmp_path / directory).is_dir() for directory in ("tmp", "cache/extensions", "cache/triton", "config"))
    assert env["PATH"].startswith(str(tmp_path / "env/bin"))
    with pytest.raises(ValueError, match="isolated environment"):
        gpu_python.prepare_environment(tmp_path, "/usr/local/cuda", {})
