from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from .yolo_service import YoloService


class YoloSignals(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    canceled = pyqtSignal()  # Сигнал отмены

class YoloWorker(QRunnable):
    def __init__(self, service: YoloService, image_paths: list[str], conf: float):
        super().__init__()
        self.service = service
        self.image_paths = image_paths
        self.conf = conf
        self.signals = YoloSignals()
        self.setAutoDelete(True)
        self._is_cancelled = False

    def cancel(self):
        """Помечает задачу на отмену (вызывается из основного потока)"""
        self._is_cancelled = True

    def run(self):
        try:
            results = []
            total = len(self.image_paths)
            for i, path in enumerate(self.image_paths):
                if self._is_cancelled:
                    self.signals.canceled.emit()
                    return  # Прерываем цикл, поток завершается безопасно

                det = self.service.infer_image(path, conf=self.conf)
                results.append({"path": path, "detections": det})
                self.signals.progress.emit(i + 1, total)

            self.signals.finished.emit(results)
        except Exception as e:
            self.signals.error.emit(str(e))
