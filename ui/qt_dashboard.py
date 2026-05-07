import argparse
import sys
from pathlib import Path

try:
    from PyQt6.QtWidgets import QApplication
    from .dashboard_window import RealTimeDashboard
    from .theme import apply_app_theme
except ImportError:
    # Support: python ui/qt_dashboard.py from the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from PyQt6.QtWidgets import QApplication
    from ui.dashboard_window import RealTimeDashboard
    from ui.theme import apply_app_theme


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="tcp://127.0.0.1:5555", help="ZMQ PUB url (default: tcp://127.0.0.1:5555)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    apply_app_theme(app)
    window = RealTimeDashboard(url=args.url)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
