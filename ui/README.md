# `ui` — Desktop monitoring (Qt dashboard)

> The Streamlit UI was removed intentionally; the desktop monitor is now the main UI surface.

## What lives here
- `qt_dashboard.py`: thin CLI entry point for launching the PyQt dashboard.
- `dashboard_window.py`: dashboard state, telemetry handling, analytics rendering, fills, PnL, and trades logic.
- `dashboard_layout.py`: widget/layout construction for all tabs.
- `theme.py`: visual-only v2 theme, table styling, card labels, and plot styling helpers.
- `axis.py`: dense trading-time axis for NAV/drawdown plots.
- `listener.py`: background ZMQ subscriber thread.

## Architecture
The dashboard is designed for **fast backtests** where the engine can emit updates much faster
than Qt can render.

### Threads
- A background `ZmqListener` thread drains the SUB socket.
- The GUI thread renders at a controlled pace.

### Flow control / performance design
- **NAV**: keep only the latest NAV per drain cycle; no per-message GUI redraw.
- **Fills**: batch fills and emit them in chunks to prevent Qt event-queue overload.
- **Rendering**:
  - NAV plot is throttled and downsampled; full horizon is retained without slow redraws.
  - Fills/trades tables update incrementally and only as needed.

## UI v2 design scope
This version is intentionally visual/modular only. It does not add new trading logic or analytics
beyond the existing dashboard behavior. The changes are mainly:
- dark navy theme instead of flat grey/black;
- card-like top metrics;
- cleaner tabs, tables, splitters, and scrollbars;
- centralized plot colors and styling;
- smaller `qt_dashboard.py` entrypoint so future UI patches are less painful.

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
