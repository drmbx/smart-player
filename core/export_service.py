import csv
import os
import cv2
from pathlib import Path


class ExportService:
    def export_to_csv(self, detections: list[dict], output_path: str, class_names: dict = None) -> str:
        """Экспорт в CSV (совместим с Excel, UTF-8)"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
            for item in detections:
                fname = Path(item['path']).name
                for det in item['detections']:
                    cls_id = det['cls']
                    cls_name = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)
                    writer.writerow([fname, cls_id, cls_name, det['conf'], *det['bbox']])
        return output_path

    def export_to_images(self, detections: list[dict], output_dir: str, class_names: dict = None,
                         progress_callback=None) -> str:
        """Экспорт изображений с отрисовкой рамок. Сохраняет оригиналы, добавляет суффикс _detected"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        total = len(detections)

        for i, item in enumerate(detections):
            if progress_callback:
                progress_callback(i, total)

            img = cv2.imread(item['path'])
            if img is None:
                continue

            for det in item['detections']:
                x1, y1, x2, y2 = det['bbox']
                color = (0, 255, 0)  # BGR
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                cls_name = class_names.get(det['cls'], str(det['cls'])) if class_names else str(det['cls'])
                label = f"{cls_name} {det['conf']:.2f}"
                # Защита от выхода текста за верхнюю границу
                text_y = max(20, y1 - 10)
                cv2.putText(img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            stem = Path(item['path']).stem
            suffix = Path(item['path']).suffix
            out_path = os.path.join(output_dir, f"{stem}_detected{suffix}")
            cv2.imwrite(out_path, img)

        if progress_callback:
            progress_callback(total, total)
        return output_dir