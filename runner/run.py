from __future__ import annotations
import argparse, asyncio, random, traceback, json
from pathlib import Path
from datetime import datetime
from runner.config import Config
from runner.engine import run_stream
from common.eventlog import EventLogger
from runner.logging_utils import setup_logging

def run_once(config_path: str, out_dir_override: str | None = None, name_override: str | None = None) -> Path:
    cfg = Config.load(config_path)

    seed = int(cfg.get("run","seed", default=0))
    if seed:
        random.seed(seed)

    out_dir = Path(out_dir_override or cfg.get("run","out_dir", default="runs"))
    name = name_override or cfg.get("run","name", default="run")

    run_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_dir, level=str(cfg.get("run","log_level", default="INFO")))
    logger.info("Run created: %s", run_dir)
    logger.info("Config path: %s", config_path)

    # Save effective merged config for reproducibility
    (run_dir / "effective_config.json").write_text(json.dumps(cfg.raw, indent=2), encoding="utf-8")

    event_logger = EventLogger(run_dir)

    try:
        asyncio.run(run_stream(cfg, run_dir, event_logger, logger_obj=logger))
        logger.info("Run completed OK: %s", run_dir)
    except Exception as e:
        logger.error("Run failed: %s", e)
        logger.error("Traceback:\n%s", traceback.format_exc())
        raise

    print(f"Run completed: {run_dir}")
    return run_dir

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Run YAML (references module YAMLs)")
    ap.add_argument("config_pos", nargs="?", help="Positional run YAML (alternative to --config)")
    ap.add_argument("--out-dir", default=None, help="Override run.out_dir from YAML")
    ap.add_argument("--name", default=None, help="Override run.name from YAML")
    args = ap.parse_args()

    config_path = args.config or args.config_pos
    if not config_path:
        ap.error("missing config file (provide --config <path> or positional config)")

    run_once(config_path, out_dir_override=args.out_dir, name_override=args.name)

if __name__ == "__main__":
    main()
