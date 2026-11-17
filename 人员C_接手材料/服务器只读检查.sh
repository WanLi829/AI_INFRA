#!/usr/bin/env bash
# 在有权限访问的 Linux 服务器执行。仅检查，不安装软件、不启动推理、不停止进程。
# bash 服务器只读检查.sh [Kairos目录] [模型snapshot目录]
set -u
export PYTHONDONTWRITEBYTECODE=1
task_root="${1:-/root/241880423/code/kairos}"
task_model="${2:-/root/241880423/models/hub2/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
printf '\n[GPU]\n'
nvidia-smi
printf '\n[内存与磁盘]\n'
free -h
df -h "$task_root" "$task_model"
printf '\n[18765–18768 端口；无匹配行表示本次未发现监听]\n'
ss -ltn | awk 'NR==1 || /:1876[5-8][[:space:]]/'
if ! cd "$task_root"; then exit 1; fi
printf '\n[源码版本与修改状态]\n'
git rev-parse HEAD
git status --short
task_python="$task_root/.venv/bin/python"
if [ ! -x "$task_python" ]; then
    printf '未找到可执行的虚拟环境 Python：%s\n' "$task_python"
    exit 1
fi
printf '\n[依赖元数据与可用 CPU；不加载模型]\n'
"$task_python" -B - <<'PY'
import importlib.metadata as m
import os
import sys
print('Python:', sys.version)
for name in ('torch', 'torchvision', 'torchaudio', 'vllm', 'lmcache', 'transformers', 'httpx', 'numpy'):
    try:
        print(name, m.version(name))
    except m.PackageNotFoundError:
        print(name, '未安装')
available = os.sched_getaffinity(0)
print('允许的 CPU:', sorted(available))
required = {4, 5, 68, 69, 6, 7, 70, 71}
print('历史 CPU 配置可用:', required <= available)
print('缺失 CPU:', sorted(required - available))
PY
printf '\n[源码、负载与模型小文件校验；不校验大权重，不启动 GPU]\n'
"$task_python" -B -m kairos.verify --model "$task_model" --metadata-only
task_status=$?
printf '\n校验退出码：%s。完整模型权重校验和 GPU 复跑尚未执行。\n' "$task_status"
exit "$task_status"
