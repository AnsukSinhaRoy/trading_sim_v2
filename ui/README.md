# `ui` — Desktop monitoring (Qt dashboard)

> The Streamlit UI was removed intentionally; we’ll revisit it later.

## What lives here
- `qt_dashboard.py`: a PyQt6 + pyqtgraph live dashboard subscribing to engine ZMQ.

## Architecture
The dashboard is designed for **fast backtests** where the engine can emit updates much faster
than Qt can render.

### Threads
- A background `ZmqListener` thread drains the SUB socket.
- The GUI thread renders at a controlled pace.

### Flow control / performance design
- **NAV**: keep only the latest NAV per drain cycle (no per-message GUI signals).
- **Fills**: batch fills and emit them in chunks (prevents Qt event-queue overload).
- **Rendering**:
  - NAV plot is throttled and downsampled (full horizon retained without slow redraws).
  - Fills/trades tables update incrementally and only as needed.

## Design choices (and why)
- **ZMQ PUB/SUB**: no direct coupling to engine process; works locally or across machines.
- **Batching + throttling**: avoids UI lag and keeps trades/fills in sync with NAV updates.
- **Separate “recent fills” window vs “full fills store”**: keeps the UI responsive while still
  retaining state for inspection.

## Running
```bash
python ui/qt_dashboard.py --url tcp://127.0.0.1:5555
```

Install optional deps via `requirements-ui.txt` or `pip install -e ".[ui]"`.


CLI override example:
```bash
python -m runner configs/run/cube_demo_ema_long.yaml --zmq-port 5560
python ui/qt_dashboard.py --url tcp://127.0.0.1:5560
```
