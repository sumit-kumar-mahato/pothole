import numpy as np

class SimpleTracker:
    def __init__(self, distance_threshold=50):
        self.objects = []
        self.threshold = distance_threshold

    def is_new(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        for obj in self.objects:
            ox, oy = obj
            dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
            if dist < self.threshold:
                return False

        self.objects.append((cx, cy))
        return True