# SFID Downstream Detection Benchmark

Frozen FINet detectors (SE-YOLOv5) evaluated on SFID foggy test images (1404 fog views).
Clear = non-fog test images (1339) provided as in-domain reference.
Restoration models are frozen; detector weights frozen; no retraining.

## mAP@0.5

| Method | YOLOv5m (clear-trained) | YOLOv5m (fog-trained) | SE-YOLOv5m (fog-trained) |
|---|------|---|---|
| Clear | 0.985 | NA | NA |
| Degraded | 0.947 | 0.993 | 0.993 |
| A5 | 0.925 | 0.991 | 0.991 |
| DehazeFormer | 0.938 | 0.989 | 0.991 |
| Restormer | 0.930 | 0.988 | 0.990 |

## mAP@0.5:0.95

| Method | YOLOv5m (clear-trained) | YOLOv5m (fog-trained) | SE-YOLOv5m (fog-trained) |
|---|------|---|---|
| Clear | 0.777 | NA | NA |
| Degraded | 0.708 | 0.838 | 0.834 |
| A5 | 0.685 | 0.806 | 0.811 |
| DehazeFormer | 0.700 | 0.808 | 0.808 |
| Restormer | 0.690 | 0.809 | 0.811 |
