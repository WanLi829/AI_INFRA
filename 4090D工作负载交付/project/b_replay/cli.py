from __future__ import annotations
import argparse
from pathlib import Path
from .common import ROOT, MODELS

def main():
    parser = argparse.ArgumentParser(description="B 数据审计与 Replay 准备")
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("acquire")
    acquire.add_argument("--a-zip", type=Path, default=ROOT.parent / "A_文献调研与负载选型_交付包_v1.0.zip")
    acquire.add_argument("--workers", type=int, default=8)
    sub.add_parser("audit")
    convert = sub.add_parser("convert")
    convert.add_argument("--model", choices=MODELS, required=True)
    convert.add_argument("--max-length", type=int, default=16384)
    export = sub.add_parser("export")
    export.add_argument("--model", choices=MODELS, required=True)
    sub.add_parser("validate")
    arrival = sub.add_parser("arrivals")
    arrival.add_argument("--bundle", type=Path, required=True)
    arrival.add_argument("--jps", type=float, required=True)
    arrival.add_argument("--seed", type=int, default=20261030)
    arrival.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument("--arrivals", type=Path, required=True)
    replay.add_argument("--endpoint", required=True)
    replay.add_argument("--model", required=True, help="Exact model identity served by the running engine")
    replay.add_argument("--method", choices=("kairos", "continuum"), required=True,
                        help="Must match the method configured in the already-running service")
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--request-timeout", type=float, default=120)
    replay.add_argument("--run-timeout", type=float, default=900)
    replay.add_argument("--mock", action="store_true", help="Label output as protocol-only, never GPU evidence")
    sub.add_parser("package")
    sub.add_parser("verify-release")
    for name in ("server-plan","server-run"):
        server=sub.add_parser(name,help="B inputs with frozen Kairos components and a resource profile")
        server.add_argument("--bundle",type=Path,required=True)
        server.add_argument("--arrivals",type=Path,required=True)
        server.add_argument("--model",required=True)
        server.add_argument("--method",choices=("kairos","continuum"),default="kairos")
        server.add_argument("--output",type=Path,required=True)
        server.add_argument("--gpu",type=int,default=0)
        server.add_argument("--server-profile",choices=("frozen","ram30"),default="frozen",
                            help="ram30: retain 8 sequences/8 GiB KV, reduce loading peak and allocator fragmentation")
        server.add_argument("--gpu-pid-policy",choices=("strict","record-only"),default="strict",
                            help="ram30 only: record-only logs unresolved GPU PIDs without rejecting the run")
        server.add_argument("--engine-cpus",default="auto")
        server.add_argument("--helper-cpus",default="auto")
        server.add_argument("--start-timeout",type=float,default=2400)
        server.add_argument("--request-timeout",type=float,default=120)
        server.add_argument("--run-timeout",type=float,default=900)
    args = parser.parse_args()
    if args.command == "acquire":
        from .acquire import bootstrap, download
        bootstrap(args.a_zip)
        download(args.workers)
    elif args.command in ("audit", "convert", "export", "validate"):
        from . import pipeline
        if args.command == "audit": pipeline.audit()
        elif args.command == "convert": pipeline.convert(args.model, args.max_length)
        elif args.command == "export": pipeline.export(args.model)
        else: pipeline.validate()
    elif args.command == "arrivals":
        from .client import make_arrivals
        make_arrivals(args.bundle, args.jps, args.seed, args.output)
    elif args.command == "replay":
        import asyncio
        from .client import replay
        asyncio.run(replay(args))
    elif args.command == "package":
        from .delivery import package
        package()
    elif args.command == "verify-release":
        import json
        from .delivery import verify_release
        print(json.dumps(verify_release()))
    elif args.command in ("server-plan","server-run"):
        import json
        from .server import make_plan,run
        if args.command=="server-plan":print(json.dumps(make_plan(args),ensure_ascii=False,indent=2))
        else:run(args)
