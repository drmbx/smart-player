from PyQt6.QtCore import QRunnable, QObject, pyqtSignal
from .export_service import ExportService

class ExportSignals(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

class ExportWorker(QRunnable):
    def __init__(self, service: ExportService, detections: list, mode: str, output_path: str, class_names: dict = None):
        super().__init__()
        self.service = service
        self.detections = detections
        self.mode = mode
        self.output_path = output_path
        self.class_names = class_names
        self.signals = ExportSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            if not self.detections:
                self.signals.error.emit("Нет детекций для экспорта")
                return

            if self.mode == 'csv':
                path = self.service.export_to_csv(self.detections, self.output_path, self.class_names)
                self.signals.finished.emit(f"✅ CSV сохранён: {path}")
            elif self.mode == 'images':
                def cb(cur, tot): self.signals.progress.emit(cur, tot)
                path = self.service.export_to_images(self.detections, self.output_path, self.class_names, cb)
                self.signals.finished.emit(f"✅ Изображения сохранены в: {path}")
        except Exception as e:
            self.signals.error.emit(f"❌ Ошибка экспорта: {str(e)}")