import os

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget, QFileDialog, QCheckBox,
)
print('loading')
from ultralytics import YOLO
print('loaded')

# from yolo_service import YoloService


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Smart Player")
        self.resize(1200, 800)
        self._build_layout()

        self._images = []
        self._current_index = 0

        self._timer = QTimer()
        self._timer.timeout.connect(self._play_step)
        self._is_playing = False
        self._btn_play.clicked.connect(self._toggle_play)

        self._btn_images_dir.clicked.connect(self._open_images_dir)
        self._btn_prev.clicked.connect(self._show_prev)
        self._btn_next.clicked.connect(self._show_next)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin_fps.valueChanged.connect(self._update_timer_interval)

        # self._yolo_service = YoloService()
        self._detections = []

    def _build_layout(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        layout.addLayout(self._build_top_buttons())
        layout.addLayout(self._build_controls())
        layout.addWidget(self._build_image_view(), stretch=1)
        layout.addLayout(self._build_slider())
        layout.addLayout(self._build_down_buttons())

    def _build_top_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._btn_images_dir = QPushButton("Папка с картинками…")
        self._btn_detect_all = QPushButton("Детекция на всех кадрах")

        row.addWidget(self._btn_images_dir)
        row.addWidget(self._btn_detect_all)

        return row

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()

        row.addWidget(QLabel("Порог уверенности:"))

        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.01, 1.0)
        self._spin_conf.setSingleStep(0.05)
        self._spin_conf.setValue(0.25)
        self._spin_conf.setDecimals(2)
        row.addWidget(self._spin_conf)

        row.addStretch()

        self._label_info = QLabel("Откройте видео или изображения")
        row.addWidget(self._label_info)

        return row

    def _build_image_view(self) -> QLabel:
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumHeight(400)
        self._image_label.setStyleSheet("background: #222; color: #888;")
        return self._image_label

    def _build_slider(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        row.addWidget(self._slider)

        self._label_frame = QLabel("0 / 0")
        row.addWidget(self._label_frame)

        return row

    def _build_down_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._btn_prev = QPushButton("◀ Кадр")
        self._btn_play = QPushButton("▶")
        self._btn_next = QPushButton("Кадр ▶")

        # FPS настройка
        row.addWidget(QLabel("FPS:"))
        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(1, 1000)
        self._spin_fps.setValue(10)
        self._spin_fps.setDecimals(0)
        self._spin_fps.setSingleStep(1)
        row.addWidget(self._spin_fps)

        # Зацикливание
        self._checkbox_loop = QCheckBox("Loop")
        row.addWidget(self._checkbox_loop)

        row.addStretch()
        row.addWidget(self._btn_prev)
        row.addWidget(self._btn_play)
        row.addWidget(self._btn_next)
        row.addStretch()

        return row

    def _open_images_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if not folder:
            return

        exts = (".png", ".jpg", ".jpeg", ".bmp")

        self._images = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith(exts)
        ]

        if not self._images:
            self._label_info.setText("Нет изображений в папке")
            return

        self._current_index = 0

        self._slider.setMaximum(len(self._images) - 1)
        self._slider.setValue(0)

        self._label_info.setText(f"Загружено изображений: {len(self._images)}")

        self._show_image()

    def _show_image(self):
        if not self._images:
            return

        path = self._images[self._current_index]

        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._label_info.setText("Ошибка загрузки изображения")
            return

        scaled = pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self._image_label.setPixmap(scaled)
        self._label_frame.setText(f"{self._current_index + 1} / {len(self._images)}")

    def _show_prev(self):
        if not self._images:
            return

        if self._current_index > 0:
            self._current_index -= 1
            self._slider.setValue(self._current_index)
            self._show_image()

    def _show_next(self):
        if not self._images:
            return

        if self._current_index < len(self._images) - 1:
            self._current_index += 1
            self._slider.setValue(self._current_index)
            self._show_image()

    def _on_slider_changed(self, value):
        if not self._images:
            return

        self._current_index = value
        self._show_image()

    def _toggle_play(self):
        if not self._images:
            return

        if self._is_playing:
            self._timer.stop()
            self._btn_play.setText("▶")
            self._is_playing = False
        else:
            fps = self._spin_fps.value()
            interval = int(1000 / fps)

            self._timer.start(interval)
            self._btn_play.setText("⏸")
            self._is_playing = True

    def _play_step(self):
        if not self._images:
            return

        if self._current_index < len(self._images) - 1:
            self._current_index += 1
        else:
            if self._checkbox_loop.isChecked():
                self._current_index = 0
            else:
                self._timer.stop()
                self._btn_play.setText("▶")
                self._is_playing = False
                return

        self._slider.setValue(self._current_index)

    def _update_timer_interval(self):
        if self._is_playing:
            fps = self._spin_fps.value()
            self._timer.setInterval(int(1000 / fps))

    def run_yolo(self):
        if not self._images:
            return

        self._label_info.setText("Запуск YOLO...")

        self._detections = self._yolo_service.infer_folder(self._images)

        self._label_info.setText("YOLO завершён")