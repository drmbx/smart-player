from ultralytics import YOLO


class YoloService:
    def __init__(self, weights_path: str = "./weights/best.pt"):
        self._model = YOLO(weights_path)

    def infer_image(self, image_path: str):
        results = self._model.predict(
            source=image_path,
            conf=0.01,
            verbose=False
        )

        r = results[0]

        detections = []

        if r.boxes is None:
            return detections

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())

            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": conf,
                "cls": cls
            })

        return detections

    def infer_folder(self, image_paths: list[str]):
        all_results = []

        for path in image_paths:
            det = self.infer_image(path)

            all_results.append({
                "path": path,
                "detections": det
            })

        return all_results
