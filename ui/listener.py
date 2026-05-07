import json
import time
from typing import List

import zmq
from PyQt6.QtCore import QThread, pyqtSignal


class ZmqListener(QThread):
    """Receives ZMQ messages on a background thread.

    Key constraints for the dashboard:
    - Qt GUI thread must stay responsive.
    - In fast backtests, per-message Qt signals can overwhelm the event queue.

    Strategy:
    - Drain the SUB socket quickly.
    - Keep only the latest NAV per drain cycle.
    - Batch FILL messages and emit them in chunks.
    """

    nav_signal = pyqtSignal(dict)
    fills_signal = pyqtSignal(list)
    learn_signal = pyqtSignal(dict)

    def __init__(self, url: str, *, max_fills_emit: int = 2000, poll_ms: int = 50, max_drain_ms: int = 15):
        super().__init__()
        self.url = url
        self._running = True
        self._max_fills_emit = int(max(1, max_fills_emit))
        self._poll_ms = int(max(1, poll_ms))
        self._max_drain_s = float(max(1, max_drain_ms)) / 1000.0

    def stop(self):
        self._running = False

    def run(self):
        ctx = zmq.Context.instance()

        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 10000)
        sock.connect(self.url)
        sock.setsockopt_string(zmq.SUBSCRIBE, "nav")
        sock.setsockopt_string(zmq.SUBSCRIBE, "fill")
        sock.setsockopt_string(zmq.SUBSCRIBE, "learn")

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        latest_nav = None
        latest_learn = None
        fills_batch: List[dict] = []

        def _decode(msg_b: bytes):
            try:
                return json.loads(msg_b.decode("utf-8"))
            except Exception as e:
                print(f"ZMQ decode error: {e}")
                return None

        def _flush():
            nonlocal latest_nav, latest_learn, fills_batch
            if latest_nav is not None:
                self.nav_signal.emit(latest_nav)
                latest_nav = None
            if latest_learn is not None:
                self.learn_signal.emit(latest_learn)
                latest_learn = None
            if fills_batch:
                # Emit in chunks to avoid huge cross-thread payloads.
                n = len(fills_batch)
                if n <= self._max_fills_emit:
                    self.fills_signal.emit(fills_batch)
                else:
                    for i in range(0, n, self._max_fills_emit):
                        self.fills_signal.emit(fills_batch[i:i + self._max_fills_emit])
                fills_batch = []

        try:
            while self._running:
                events = dict(poller.poll(self._poll_ms))

                # Even if no new events, flush pending batches (rare).
                if sock not in events:
                    _flush()
                    continue

                # Drain all available messages quickly, but time-bound.
                start = time.perf_counter()
                while True:
                    if (time.perf_counter() - start) >= self._max_drain_s:
                        break
                    try:
                        topic_b, msg_b = sock.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break

                    data = _decode(msg_b)
                    if data is None:
                        continue

                    topic = topic_b.decode("utf-8", errors="ignore")
                    if topic == "nav":
                        latest_nav = data
                    elif topic == "fill":
                        fills_batch.append(data)
                    elif topic == "learn":
                        latest_learn = data

                _flush()
        finally:
            try:
                sock.close(0)
            except Exception:
                pass


