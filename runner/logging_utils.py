from __future__ import annotations
import logging
from pathlib import Path
import sys

def setup_logging(run_dir: Path, level: str = "INFO") -> logging.Logger:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    logger = logging.getLogger("levitate")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Avoid duplicate handlers if called twice
    if not any(isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path for h in logger.handlers):
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger
