import os
import sys

from PyQt6.QtWidgets import QApplication

# Добавляем корень проекта в PYTHONPATH для корректных импортов
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
