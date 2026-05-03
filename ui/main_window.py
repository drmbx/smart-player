import os
from datetime import datetime
from PyQt6.QtCore import Qt, QTimer, QThreadPool
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSlider, QVBoxLayout, QWidget, QFileDialog, QCheckBox, QProgressBar, QMessageBox
)

from core.yolo_service import YoloService
from core.yolo_worker import YoloWorker
from core.export_service import ExportService
from core.export_worker import ExportWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Smart Player")
        self.resize(1200, 800)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.hide()

        self._build_layout()

        self._images = []
        self._current_index = 0
        self._detections = []
        self._active_worker = None

        self._timer = QTimer()
        self._timer.timeout.connect(self._play_step)
        self._is_playing = False

        # Подключение сигналов
        self._btn_play.clicked.connect(self._toggle_play)
        self._btn_images_dir.clicked.connect(self._open_images_dir)

        self._btn_prev.clicked.connect(self._show_prev)
        self._btn_next.clicked.connect(self._show_next)

        # Новые кнопки навигации по объектам
        self._btn_prev_detected.clicked.connect(self._show_prev_detected)
        self._btn_next_detected.clicked.connect(self._show_next_detected)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin_fps.valueChanged.connect(self._update_timer_interval)

        self._btn_detect_all.clicked.connect(self._start_detection)
        self._btn_cancel_detect.clicked.connect(self._cancel_detection)

        self._btn_export_csv.clicked.connect(self._start_export_csv)
        self._btn_export_imgs.clicked.connect(self._start_export_images)

        # Сервисы
        self._yolo_service = YoloService()
        self._export_service = ExportService()
        self._thread_pool = QThreadPool.globalInstance()
        self._thread_pool.setMaxThreadCount(4)

        self._update_export_buttons(False)
        self._update_nav_buttons()  # Инициализация состояния кнопок

    def _build_layout(self) -> None:
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
        self._btn_images_dir = QPushButton("📁 Папка с картинками…")

        self._btn_detect_all = QPushButton("🔍 Детекция")
        self._btn_cancel_detect = QPushButton("❌ Отмена")
        self._btn_cancel_detect.hide()

        self._btn_export_csv = QPushButton("📄 Экспорт CSV")
        self._btn_export_imgs = QPushButton("🖼️ Экспорт картинок")
        self._btn_export_csv.setEnabled(False)
        self._btn_export_imgs.setEnabled(False)

        row.addWidget(self._btn_images_dir, stretch=1)
        row.addWidget(self._btn_detect_all, stretch=1)
        row.addWidget(self._btn_cancel_detect, stretch=1)
        row.addWidget(self._btn_export_csv, stretch=1)
        row.addWidget(self._btn_export_imgs, stretch=1)
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

        row.addWidget(self._progress_bar, stretch=1)

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

        # Кнопки навигации по объектам
        self._btn_prev_detected = QPushButton("◀ Объект")
        self._btn_next_detected = QPushButton("Объект ▶")

        # Обычные кнопки навигации
        self._btn_prev = QPushButton("◀ Кадр")
        self._btn_play = QPushButton("▶")
        self._btn_next = QPushButton("Кадр ▶")

        row.addWidget(QLabel("FPS:"))
        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(1, 1000)
        self._spin_fps.setValue(10)
        self._spin_fps.setDecimals(0)
        self._spin_fps.setSingleStep(1)
        row.addWidget(self._spin_fps)

        self._checkbox_loop = QCheckBox("Loop")
        row.addWidget(self._checkbox_loop)

        row.addStretch()

        # Компоновка: [Объект ◀] [◀ Кадр] [▶] [Кадр ▶] [Объект ▶]
        row.addWidget(self._btn_prev_detected)
        row.addWidget(self._btn_prev)
        row.addWidget(self._btn_play)
        row.addWidget(self._btn_next)
        row.addWidget(self._btn_next_detected)

        row.addStretch()
        return row

    def _update_export_buttons(self, enabled: bool) -> None:
        self._btn_export_csv.setEnabled(enabled)
        self._btn_export_imgs.setEnabled(enabled)

    def _update_nav_buttons(self) -> None:
        """Обновляет состояние всех кнопок навигации"""
        has_images = bool(self._images)
        has_detections = bool(self._detections)

        # Обычные кнопки
        self._btn_prev.setEnabled(has_images and self._current_index > 0)
        self._btn_next.setEnabled(has_images and self._current_index < len(self._images) - 1)

        # Кнопки перехода по объектам
        if has_detections:
            self._btn_prev_detected.setEnabled(self._find_prev_detected_index(self._current_index) is not None)
            self._btn_next_detected.setEnabled(self._find_next_detected_index(self._current_index) is not None)
        else:
            self._btn_prev_detected.setEnabled(False)
            self._btn_next_detected.setEnabled(False)

    def _toggle_detection_buttons(self, running: bool) -> None:
        if running:
            self._btn_detect_all.hide()
            self._btn_cancel_detect.show()
            self._btn_images_dir.setEnabled(False)
        else:
            self._btn_cancel_detect.hide()
            self._btn_detect_all.show()
            self._btn_images_dir.setEnabled(True)

    def _open_images_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if not folder: return

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
        self._detections = []
        self._slider.setMaximum(len(self._images) - 1)
        self._slider.setValue(0)
        self._label_info.setText(f"Загружено изображений: {len(self._images)}")
        self._update_export_buttons(False)
        self._show_image()

    def _show_image(self):
        if not self._images: return
        path = self._images[self._current_index]
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._label_info.setText("Ошибка загрузки изображения")
            return

        det_for_img = self._get_detections_for_current_image()
        if det_for_img:
            self._draw_detections(pixmap, det_for_img)

        scaled = pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
        self._label_frame.setText(f"{self._current_index + 1} / {len(self._images)}")

        # Обновляем состояние кнопок после отображения кадра
        self._update_nav_buttons()

    def _get_detections_for_current_image(self) -> list[dict]:
        if not self._images or not self._detections: return []
        path = self._images[self._current_index]
        return next((item["detections"] for item in self._detections if item["path"] == path), [])

    def _draw_detections(self, pixmap: QPixmap, detections: list[dict]) -> None:
        painter = QPainter(pixmap)
        pen = QPen(QColor(0, 255, 0), 2)
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            painter.setPen(pen)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            label = f"{det['cls']} {det['conf']:.2f}"
            painter.setPen(QPen(QColor(0, 255, 0)))
            painter.fillRect(x1, y1 - 18, painter.fontMetrics().horizontalAdvance(label) + 6, 18, QColor(0, 255, 0))
            painter.setPen(QPen(Qt.GlobalColor.black))
            painter.drawText(x1 + 3, y1 - 4, label)
        painter.end()

    def _has_detections_at(self, index: int) -> bool:
        if 0 <= index < len(self._detections):
            return bool(self._detections[index]["detections"])
        return False

    def _find_next_detected_index(self, current: int) -> int | None:
        for i in range(current + 1, len(self._images)):
            if self._has_detections_at(i): return i
        return None

    def _find_prev_detected_index(self, current: int) -> int | None:
        for i in range(current - 1, -1, -1):
            if self._has_detections_at(i): return i
        return None

    def _show_prev(self):
        if not self._images or self._current_index <= 0: return
        self._current_index -= 1
        self._slider.setValue(self._current_index)
        self._show_image()

    def _show_next(self):
        if not self._images or self._current_index >= len(self._images) - 1: return
        self._current_index += 1
        self._slider.setValue(self._current_index)
        self._show_image()

    def _show_prev_detected(self):
        idx = self._find_prev_detected_index(self._current_index)
        if idx is not None:
            self._current_index = idx
            self._slider.setValue(self._current_index)
            self._show_image()

    def _show_next_detected(self):
        idx = self._find_next_detected_index(self._current_index)
        if idx is not None:
            self._current_index = idx
            self._slider.setValue(self._current_index)
            self._show_image()

    def _on_slider_changed(self, value):
        if not self._images: return
        self._current_index = value
        self._show_image()

    def _toggle_play(self):
        if not self._images: return
        if self._is_playing:
            self._timer.stop()
            self._btn_play.setText("▶")
            self._is_playing = False
        else:
            self._timer.start(int(1000 / self._spin_fps.value()))
            self._btn_play.setText("⏸")
            self._is_playing = True

    def _play_step(self):
        if not self._images: return
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
            self._timer.setInterval(int(1000 / self._spin_fps.value()))

    def _start_detection(self):
        if not self._images or self._active_worker is not None:
            return

        conf = self._spin_conf.value()
        self._label_info.setText("Запуск YOLO...")
        self._progress_bar.setValue(0)
        self._progress_bar.setMaximum(len(self._images))
        self._progress_bar.show()

        self._toggle_detection_buttons(True)

        worker = YoloWorker(self._yolo_service, self._images, conf)
        worker.signals.progress.connect(self._on_detection_progress)
        worker.signals.finished.connect(self._on_detection_finished)
        worker.signals.error.connect(self._on_detection_error)
        worker.signals.canceled.connect(self._on_detection_canceled)

        self._active_worker = worker
        self._thread_pool.start(worker)

    def _cancel_detection(self):
        if self._active_worker:
            self._active_worker.cancel()
            self._label_info.setText("Отмена...")

    def _on_detection_progress(self, current: int, total: int):
        self._progress_bar.setValue(current)
        self._label_info.setText(f"Обработано: {current} / {total}")

    def _on_detection_finished(self, results: list):
        self._detections = results
        obj_count = sum(1 for r in results if r['detections'])
        self._label_info.setText(f"✅ Завершён. Найдено объектов в {obj_count} кадрах.")
        self._active_worker = None
        self._finalize_detection_ui()
        self._update_export_buttons(True)
        self._show_image()

    def _on_detection_error(self, msg: str):
        self._label_info.setText(f"❌ Ошибка: {msg}")
        self._active_worker = None
        self._finalize_detection_ui()
        self._update_export_buttons(False)

    def _on_detection_canceled(self):
        self._detections = []
        self._label_info.setText("Детекция отменена")
        self._active_worker = None
        self._finalize_detection_ui()
        self._update_export_buttons(False)
        self._show_image()

    def _finalize_detection_ui(self):
        self._progress_bar.hide()
        self._toggle_detection_buttons(False)

    def _start_export_csv(self):
        if not self._detections: return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", "detections.csv", "CSV Files (*.csv)")
        if not path: return
        self._run_export('csv', path)

    def _start_export_images(self):
        if not self._detections: return
        base_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f"detected_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        self._run_export('images', output_dir)

    def _run_export(self, mode: str, output_path: str):
        if mode == 'images':
            detections_to_export = [d for d in self._detections if d['detections']]
            if not detections_to_export:
                QMessageBox.information(self, "Информация", "Нет изображений с обнаруженными объектами.")
                return
        else:
            detections_to_export = self._detections

        self._btn_export_csv.setEnabled(False)
        self._btn_export_imgs.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setMaximum(len(detections_to_export))
        self._progress_bar.show()
        self._label_info.setText("Экспорт...")

        class_names = getattr(self._yolo_service._model, 'names', {})
        worker = ExportWorker(self._export_service, detections_to_export, mode, output_path, class_names)
        worker.signals.progress.connect(self._on_export_progress)
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.error.connect(self._on_export_error)

        self._thread_pool.start(worker)

    def _on_export_progress(self, current: int, total: int):
        self._progress_bar.setValue(current)
        self._label_info.setText(f"Экспорт: {current}/{total}")

    def _on_export_finished(self, message: str):
        self._label_info.setText(message)
        self._progress_bar.hide()
        self._btn_export_csv.setEnabled(True)
        self._btn_export_imgs.setEnabled(True)
        QMessageBox.information(self, "Успех", f"{message}")

    def _on_export_error(self, error: str):
        self._label_info.setText(error)
        self._progress_bar.hide()
        self._btn_export_csv.setEnabled(True)
        self._btn_export_imgs.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", error)