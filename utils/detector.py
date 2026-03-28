from ultralytics import YOLO
import cv2
import time

class PotholeDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

        self.class_names = [
            "longitudinal_crack",
            "transverse_crack",
            "alligator_crack",
            "pothole"
        ]

    # 🔥 Severity Function
    def get_severity(self, bbox):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        if area < 5000:
            return "Low"
        elif area < 15000:
            return "Medium"
        else:
            return "High"

    def detect(self, frame):
        start_time = time.time()

        results = self.model(frame, conf=0.3)  # slightly stricter

        detections = []
        stats = {cls: 0 for cls in self.class_names}

        for r in results:

            # 🔥 handle empty detection safely
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = self.class_names[int(cls)]

                stats[label] += 1

                severity = None
                if label == "pothole":
                    severity = self.get_severity((x1, y1, x2, y2))

                detections.append({
                    "label": label,
                    "confidence": float(score),
                    "bbox": (x1, y1, x2, y2),
                    "severity": severity
                })

                # 🎨 Color coding
                color = (0, 0, 255) if label == "pothole" else (255, 0, 0)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label
                cv2.putText(frame,
                            f"{label} {score:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2)

                # 🔥 Severity display
                if severity:
                    cv2.putText(frame,
                                f"{severity}",
                                (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2)

        # safer FPS
        fps = round(1 / max(time.time() - start_time, 1e-6), 2)

        return frame, detections, stats, fps