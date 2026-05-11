import os
from datetime import datetime

from PyQt6.QtCore import Qt, QThreadPool, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.export_service import ExportService
from core.export_worker import ExportWorker
from core.yolo_service import YoloService
from core.yolo_worker import YoloWorker
from ui.image_view import ImageView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Smart Player")
        self.resize(1200, 800)

        # Состояние приложения
        self._weights_path = "weights/best.pt"
        self._images = []
        self._current_index = 0
        self._detections = []  # Хранит ВСЕ детекции (conf >= 0.01)
        self._active_worker = None
        self._is_detection_running = False
        self._is_playing = False

        self._timer = QTimer()
        self._timer.timeout.connect(self._play_step)

        # UI-элементы
        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.hide()

        self._build_layout()

        # Дефолтный текст кнопки весов
        self._btn_select_weights.setText(f"{os.path.basename(self._weights_path)}")

        # Инициализация сервисов (строго один раз)
        self._yolo_service = YoloService(self._weights_path)
        self._export_service = ExportService()
        self._thread_pool = QThreadPool.globalInstance()
        self._thread_pool.setMaxThreadCount(4)

        # Подключение сигналов
        self._btn_play.clicked.connect(self._toggle_play)
        self._btn_images_dir.clicked.connect(self._open_images_dir)
        self._btn_prev.clicked.connect(self._show_prev)
        self._btn_next.clicked.connect(self._show_next)
        self._btn_prev_detected.clicked.connect(self._show_prev_detected)
        self._btn_next_detected.clicked.connect(self._show_next_detected)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin_fps.valueChanged.connect(self._update_timer_interval)

        # 🔑 Динамическая фильтрация по порогу уверенности
        self._spin_conf.valueChanged.connect(self._on_confidence_changed)

        self._btn_detect_all.clicked.connect(self._start_detection)
        self._btn_cancel_detect.clicked.connect(self._cancel_detection)
        self._btn_export_csv.clicked.connect(self._start_export_csv)
        self._btn_export_imgs.clicked.connect(self._start_export_images)
        self._image_view.zoom_changed.connect(self._on_image_zoom_changed)

        # Инициализация UI
        self._update_ui_state()

    def _update_ui_state(self) -> None:
        """Централизованное управление состоянием всех кнопок и элементов UI"""
        if not hasattr(self, "_btn_detect_all"):
            return

        has_images = bool(self._images)
        has_detections = bool(self._detections)

        # 1. Верхняя панель
        self._btn_detect_all.setEnabled(has_images and not self._is_detection_running)
        self._btn_images_dir.setEnabled(not self._is_detection_running)
        self._btn_export_csv.setEnabled(has_detections)
        self._btn_export_imgs.setEnabled(has_detections)

        # 2. Переключение Детекция ↔ Отмена
        if self._is_detection_running:
            self._btn_detect_all.hide()
            self._btn_cancel_detect.show()
            self._btn_cancel_detect.setEnabled(True)
        else:
            self._btn_cancel_detect.hide()
            self._btn_detect_all.show()
            self._btn_cancel_detect.setEnabled(False)

        # 3. Воспроизведение и слайдер
        self._btn_play.setEnabled(has_images)
        self._slider.setEnabled(has_images)

        # 4. Навигация (учитывает текущий порог уверенности)
        if has_images:
            self._btn_prev.setEnabled(self._current_index > 0)
            self._btn_next.setEnabled(self._current_index < len(self._images) - 1)

            if has_detections:
                prev_idx = self._find_prev_detected_index(self._current_index)
                next_idx = self._find_next_detected_index(self._current_index)
                self._btn_prev_detected.setEnabled(prev_idx is not None)
                self._btn_next_detected.setEnabled(next_idx is not None)
            else:
                self._btn_prev_detected.setEnabled(False)
                self._btn_next_detected.setEnabled(False)
        else:
            self._btn_prev.setEnabled(False)
            self._btn_next.setEnabled(False)
            self._btn_prev_detected.setEnabled(False)
            self._btn_next_detected.setEnabled(False)

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
        self._btn_images_dir = QPushButton("📁 Папка с изображениями…")
        self._btn_detect_all = QPushButton("🔍 Детекция")
        self._btn_cancel_detect = QPushButton("❌ Отмена")
        self._btn_cancel_detect.hide()
        self._btn_export_csv = QPushButton("📄 Экспорт CSV")
        self._btn_export_imgs = QPushButton("🖼️ Экспорт изображений")

        row.addWidget(self._btn_images_dir, stretch=1)
        row.addWidget(self._btn_detect_all, stretch=1)
        row.addWidget(self._btn_cancel_detect, stretch=1)
        row.addWidget(self._btn_export_csv, stretch=1)
        row.addWidget(self._btn_export_imgs, stretch=1)
        return row

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._btn_select_weights = QPushButton("Выбрать веса")
        self._btn_select_weights.clicked.connect(self._select_weights)
        row.addWidget(self._btn_select_weights)

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

    def _build_image_view(self) -> ImageView:
        self._image_view = ImageView()
        self._image_view.setMinimumHeight(400)
        self._image_view.setStyleSheet("background: #222;")
        return self._image_view

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
        row.setContentsMargins(0, 0, 0, 0)

        left_layout = QHBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("FPS:"))
        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(1, 1000)
        self._spin_fps.setValue(10)
        self._spin_fps.setDecimals(0)
        self._spin_fps.setSingleStep(1)
        left_layout.addWidget(self._spin_fps)
        self._checkbox_loop = QCheckBox("Loop")
        left_layout.addWidget(self._checkbox_loop)
        left_layout.addStretch(1)

        center_layout = QHBoxLayout()
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)
        self._btn_prev_detected = QPushButton("◀ Объект")
        self._btn_next_detected = QPushButton("Объект ▶")
        self._btn_prev = QPushButton("◀ Кадр")
        self._btn_play = QPushButton("▶")
        self._btn_next = QPushButton("Кадр ▶")
        center_layout.addWidget(self._btn_prev_detected)
        center_layout.addWidget(self._btn_prev)
        center_layout.addWidget(self._btn_play)
        center_layout.addWidget(self._btn_next)
        center_layout.addWidget(self._btn_next_detected)

        right_layout = QHBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addStretch(1)
        self._btn_reset_zoom = QPushButton("↩️ 100%")
        self._btn_reset_zoom.clicked.connect(self._reset_zoom)
        right_layout.addWidget(self._btn_reset_zoom)

        self._slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self._slider_zoom.setMinimum(10)
        self._slider_zoom.setMaximum(1000)
        self._slider_zoom.setValue(100)
        self._slider_zoom.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_zoom.setTickInterval(100)
        self._slider_zoom.setMinimumWidth(120)
        self._slider_zoom.setMaximumWidth(300)
        self._slider_zoom.valueChanged.connect(self._on_zoom_slider_changed)
        right_layout.addWidget(self._slider_zoom)

        self._label_zoom = QLabel("100%")
        self._label_zoom.setFixedWidth(50)
        self._label_zoom.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(self._label_zoom)

        row.addLayout(left_layout, 1)
        row.addLayout(center_layout)
        row.addLayout(right_layout, 1)
        return row

    def _reset_zoom(self) -> None:
        self._image_view.reset_view()

    # ------------------ ЗАГРУЗКА И НАВИГАЦИЯ ------------------

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
            self._update_ui_state()
            return

        self._current_index = 0
        self._detections = []
        self._slider.setMaximum(len(self._images) - 1)
        self._slider.setValue(0)
        self._label_info.setText(f"Загружено изображений: {len(self._images)}")
        self._update_ui_state()
        self._show_image()

    def _show_image(self) -> None:
        if not self._images: return

        path = self._images[self._current_index]
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._label_info.setText("Ошибка загрузки изображения")
            return

        # 🔑 Используем отфильтрованные по текущему порогу детекции
        det_for_img = self._get_filtered_detections_for_current_image()
        self._image_view.set_pixmap(pixmap)
        self._image_view.set_detections(det_for_img)
        self._label_frame.setText(f"{self._current_index + 1} / {len(self._images)}")
        self._update_ui_state()

    def _get_filtered_detections_for_current_image(self) -> list[dict]:
        """Возвращает детекции для текущего кадра, отфильтрованные по spin_conf"""
        if not self._images or not self._detections:
            return []
        threshold = self._spin_conf.value()
        path = self._images[self._current_index]
        raw_dets = next((item["detections"] for item in self._detections if item["path"] == path), [])
        return [det for det in raw_dets if det["conf"] >= threshold]

    def _has_detections_at(self, index: int) -> bool:
        """Проверяет наличие детекций выше порога уверенности на указанном индексе"""
        if 0 <= index < len(self._detections):
            threshold = self._spin_conf.value()
            return any(det["conf"] >= threshold for det in self._detections[index]["detections"])
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

    # 🔑 Мгновенное обновление при изменении порога
    def _on_confidence_changed(self) -> None:
        if self._detections:
            # Подсчитываем, сколько кадров имеют детекции выше текущего порога
            threshold = self._spin_conf.value()
            count_with_detections = sum(
                1 for item in self._detections
                if any(det["conf"] >= threshold for det in item["detections"])
            )

            total_objects = sum(
                1 for item in self._detections
                for det in item["detections"]
                if det["conf"] >= threshold
            )

            self._label_info.setText(
                f"Порог {threshold:.2f}: {total_objects} объектов в {count_with_detections} кадрах"
            )
            self._show_image()  # Перерисует текущий кадр с новым фильтром

    def _count_detections_above_threshold(self, threshold: float) -> tuple[int, int]:
        count_frames = 0
        count_objects = 0
        for item in self._detections:
            filtered = [det for det in item["detections"] if det["conf"] >= threshold]
            if filtered:
                count_frames += 1
                count_objects += len(filtered)
        return count_frames, count_objects

    # ------------------ ВОСПРОИЗВЕДЕНИЕ ------------------

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
        elif self._checkbox_loop.isChecked():
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

    # ------------------ ДЕТЕКЦИЯ ------------------

    def _start_detection(self):
        if not self._images or self._active_worker is not None:
            return

        self._is_detection_running = True
        # 🔑 YOLO всегда запускается с минимальным порогом, чтобы сохранить все сырые данные
        conf = 0.01
        self._label_info.setText("Запуск YOLO...")
        self._progress_bar.setValue(0)
        self._progress_bar.setMaximum(len(self._images))
        self._progress_bar.show()
        self._update_ui_state()

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
        self._is_detection_running = False
        obj_count = sum(1 for r in results if r["detections"])
        self._label_info.setText(f"✅ Завершён. Найдено объектов в {obj_count} кадрах.")
        self._active_worker = None
        self._progress_bar.hide()
        self._update_ui_state()
        self._show_image()

    def _on_detection_error(self, msg: str):
        self._is_detection_running = False
        self._label_info.setText(f"❌ Ошибка: {msg}")
        self._active_worker = None
        self._progress_bar.hide()
        self._update_ui_state()

    def _on_detection_canceled(self):
        self._is_detection_running = False
        self._detections = []
        self._label_info.setText("Детекция отменена")
        self._active_worker = None
        self._progress_bar.hide()
        self._update_ui_state()
        self._show_image()

    # ------------------ ЭКСПОРТ ------------------

    def _start_export_csv(self):
        if not self._detections: return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", "detections.csv", "CSV Files (*.csv)")
        if not path: return
        self._run_export("csv", path)

    def _start_export_images(self):
        if not self._detections: return
        base_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f"detected_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        self._run_export("images", output_dir)

    def _run_export(self, mode: str, output_path: str):
        threshold = self._spin_conf.value()

        # 🔑 Фильтруем ВСЕ детекции по текущему порогу перед экспортом
        filtered_detections = []
        for item in self._detections:
            filtered_dets = [det for det in item["detections"] if det["conf"] >= threshold]
            if filtered_dets:
                filtered_detections.append({"path": item["path"], "detections": filtered_dets})

        if not filtered_detections:
            QMessageBox.information(self, "Информация", "Нет объектов, соответствующих порогу уверенности.")
            return

        self._btn_export_csv.setEnabled(False)
        self._btn_export_imgs.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setMaximum(len(filtered_detections))
        self._progress_bar.show()
        self._label_info.setText("Экспорт...")

        class_names = getattr(self._yolo_service._model, "names", {})
        worker = ExportWorker(self._export_service, filtered_detections, mode, output_path, class_names)
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

    def _on_zoom_slider_changed(self, value: int) -> None:
        factor = value / 100.0
        self._image_view.set_zoom_factor(factor)
        self._label_zoom.setText(f"{value}%")

    def _on_image_zoom_changed(self, factor: float) -> None:
        percent = int(factor * 100)
        self._slider_zoom.blockSignals(True)
        self._slider_zoom.setValue(percent)
        self._slider_zoom.blockSignals(False)
        self._label_zoom.setText(f"{percent}%")

    def _select_weights(self) -> None:
        weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")
        if not os.path.isdir(weights_dir):
            weights_dir = os.getcwd()

        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл весов", weights_dir, "YOLO Weights (*.pt *.pth)"
        )
        if not path: return

        try:
            self._yolo_service.load_weights(path)
            self._weights_path = path
            self._btn_select_weights.setText(f"{os.path.basename(path)}")
            self._label_info.setText(f"✅ Веса загружены: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки весов", str(e))