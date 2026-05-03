from PyQt6.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap, QTransform, QWheelEvent
from PyQt6.QtWidgets import QWidget


class ImageView(QWidget):
    zoom_changed = pyqtSignal(float)  # Сигнал для синхронизации с UI

    def __init__(self):
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._detections: list[dict] = []
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self._is_panning = False
        self._last_mouse_pos = QPoint()

    def set_pixmap(self, px: QPixmap) -> None:
        self._pixmap = px
        # ✅ Убрали self._reset_view() отсюда. Зум и позиция теперь сохраняются.
        self.update()

    def reset_view(self) -> None:
        """Явный сброс зума и позиции (можно повесить на кнопку или хоткей в будущем)"""
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self.zoom_changed.emit(self._zoom_factor)
        self.update()

    def set_detections(self, detections: list[dict]) -> None:
        self._detections = detections
        self.update()

    def set_zoom_factor(self, factor: float) -> None:
        """Устанавливает зум извне (например, из слайдера)"""
        self._zoom_factor = max(0.1, min(10.0, factor))
        self.update()
        self.zoom_changed.emit(self._zoom_factor)

    def _reset_view(self) -> None:
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self.zoom_changed.emit(self._zoom_factor)

    def _get_base_transform(self) -> QTransform:
        if not self._pixmap or self._pixmap.isNull():
            return QTransform()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        return QTransform().translate((ww - pw * scale) / 2, (wh - ph * scale) / 2).scale(scale, scale)

    def paintEvent(self, event) -> None:
        if not self._pixmap or self._pixmap.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        transform = self._get_base_transform()
        transform.translate(self.width() / 2, self.height() / 2)
        transform.scale(self._zoom_factor, self._zoom_factor)
        transform.translate(-self.width() / 2 + self._pan_offset.x(),
                            -self.height() / 2 + self._pan_offset.y())

        painter.setTransform(transform)
        painter.drawPixmap(0, 0, self._pixmap)

        if self._detections:
            pen = QPen(QColor(0, 255, 0), max(2.0, 2.0 / self._zoom_factor))
            font = painter.font()
            font.setPointSize(max(10, int(10 / self._zoom_factor)))
            painter.setFont(font)

            for det in self._detections:
                x1, y1, x2, y2 = det["bbox"]
                w, h = x2 - x1, y2 - y1

                painter.setPen(pen)
                # 🔑 Сбрасываем кисть, чтобы drawRect рисовал только контур
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRectF(x1, y1, w, h))

                display_cls = det.get("cls_name", str(det.get("cls", "?")))
                label = f"{display_cls} {det['conf']:.2f}"

                bg_h = max(14, 14 / self._zoom_factor)
                painter.setPen(QPen(Qt.GlobalColor.transparent))
                painter.setBrush(QColor(0, 255, 0))
                painter.drawRect(QRectF(x1, y1 - bg_h - 2,
                                        painter.fontMetrics().horizontalAdvance(label) + 6, bg_h))
                painter.setPen(QPen(Qt.GlobalColor.black))
                painter.drawText(x1 + 3, y1 - 4, label)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() > 0:
            self._zoom_factor = min(10.0, self._zoom_factor * 1.1)
        else:
            self._zoom_factor = max(0.1, self._zoom_factor / 1.1)
        self.update()
        self.zoom_changed.emit(self._zoom_factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = True
            self._last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_panning:
            delta = event.position().toPoint() - self._last_mouse_pos
            self._pan_offset += delta
            self._last_mouse_pos = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pixmap:
            self.update()
