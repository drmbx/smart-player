from ultralytics import YOLO

class YoloService:
    def __init__(self, weights_path: str = "./weights/best.pt"):
        self._model = YOLO(weights_path)

    def infer_image(self, image_path: str, conf: float = 0.25) -> list[dict]:
        results = self._model.predict(source=image_path, conf=conf, verbose=False)
        r = results[0]
        detections = []
        if r.boxes is None:
            return detections

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_val = float(box.conf[0].cpu().numpy())
            cls_val = int(box.cls[0].cpu().numpy())
            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": conf_val,
                "cls": cls_val
            })
        return detections