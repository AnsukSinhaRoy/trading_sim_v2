from pathlib import Path
from runner.config import Config

def test_config_merge():
    cfg = Config.load("configs/run/demo_synth.yaml")
    assert cfg.get("market_feed","type") == "synthetic_1m"
    assert cfg.get("execution","initial_cash") == 1000000
    assert cfg.get("strategy","type") == "toy_rebalance"
