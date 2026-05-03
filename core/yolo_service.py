import torch
from ultralytics import YOLO


class YoloService:
    def __init__(self, weights_path: str = "./weights/best.pt"):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = YOLO(weights_path)

    def infer_image(self, image_path: str, conf: float = 0.25) -> list[dict]:
        results = self._model.predict(
            source=image_path,
            conf=conf,
            device=self._device,
            verbose=False,
        )

        r = results[0]
        detections = []

        if r.boxes is None:
            return detections

        # Получаем словарь {class_id: "class_name"} из модели
        class_names = getattr(self._model, "names", {})

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_val = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            # Берём имя класса, если нет → fallback на ID
            cls_name = class_names.get(cls_id, str(cls_id))

            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": conf_val,
                "cls": cls_id,  # оставляем для внутренней логики (навигация)
                "cls_name": cls_name,  # новое поле для UI
            })

        return detections
