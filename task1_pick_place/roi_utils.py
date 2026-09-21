"""Small geometry helpers used by the pipeline."""

from dataclasses import dataclass


@dataclass
class ROI:
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_bbox(cls, bbox, padding_ratio: float = 0.35):
        """Build a configurable ROI around an initial detection bbox, padded outward
        so a hand reaching toward the backpack is counted as 'entering' before it
        literally touches the box edge."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = w * padding_ratio, h * padding_ratio
        return cls(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def contains_center(self, bbox) -> bool:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return self.contains_point(cx, cy)

    def as_int_tuple(self):
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance(box_a, box_b) -> float:
    ax = (box_a[0] + box_a[2]) / 2
    ay = (box_a[1] + box_a[3]) / 2
    bx = (box_b[0] + box_b[2]) / 2
    by = (box_b[1] + box_b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
