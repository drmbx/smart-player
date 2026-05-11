import os
import torch
from ultralytics import YOLO

class YoloService:
    def __init__(self, weights_path: str = "weights/best.pt"):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[YOLO] Инициализация на устройстве: {self._device.upper()}")
        self._model = None
        self.load_weights(weights_path)

    def load_weights(self, weights_path: str) -> None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл весов не найден: {weights_path}")
        self._model = YOLO(weights_path)
        print(f"[YOLO] Загружены веса: {weights_path}")

    def infer_image(self, image_path: str, conf: float = 0.25) -> list[dict]:
        results = self._model.predict(
            source=image_path,
            conf=conf,
            device=self._device,
            verbose=False
        )

        r = results[0]
        detections = []

        if r.boxes is None:
            return detections

        class_names = getattr(self._model, 'names', {})

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_val = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = class_names.get(cls_id, str(cls_id))

            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": conf_val,
                "cls": cls_id,
                "cls_name": cls_name
            })

        return detections