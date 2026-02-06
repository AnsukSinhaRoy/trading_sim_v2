from __future__ import annotations
import argparse
import sys
import subprocess
from pathlib import Path

import yaml

from runner.run import run_once
from preprocess.build import run_preprocess

def _detach_run(config_path: str, out_dir: str | None, name: str | None) -> None:
    args = [sys.executable, "-u", "-m", "runner", config_path]
    if out_dir:
        args += ["--out-dir", out_dir]
    if name:
        args += ["--name", name]

    kwargs = {}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True

    subprocess.Popen(args, **kwargs)
    print("Detached: started background run.")
    print("Tip: check the latest folder under ./runs/ and open runs/<run_id>/run.log")

def _is_preprocess_yaml(path: str) -> bool:
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return isinstance(data, dict) and ("preprocess" in data or data.get("task") == "preprocess")
    except Exception:
        return False

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="levitate",
        description="Run a trading experiment OR a preprocess job from a config YAML.",
    )

    ap.add_argument("config", nargs="?", help="YAML path, e.g. configs/run/demo_synth.yaml")
    ap.add_argument("--config", dest="config_flag", default=None, help="Alias for positional config (for compatibility).")
    ap.add_argument("--out-dir", default=None, help="Override run.out_dir from YAML (run mode only)")
    ap.add_argument("--name", default=None, help="Override run.name from YAML (run mode only)")
    ap.add_argument("--detach", action="store_true", help="Run mode only: run in background (spawns a new console/process).")

    args = ap.parse_args()
    cfg_path = args.config_flag or args.config
    if not cfg_path:
        ap.error("missing config file (provide positional config or --config <path>)")

    if _is_preprocess_yaml(cfg_path):
        # Preprocess is always foreground for now (it's a batch job)
        run_preprocess(cfg_path)
        return

    if args.detach:
        _detach_run(cfg_path, args.out_dir, args.name)
        return

    run_once(cfg_path, out_dir_override=args.out_dir, name_override=args.name)

if __name__ == "__main__":
    main()
