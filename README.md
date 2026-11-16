# LiDAR Intersection Perception — Educational Mini Project

A reproducible **software-only** Python/PyTorch prototype of 3D road-intersection perception. It creates synthetic XYZI points, trains a small pointwise MLP to classify **ground, pedestrians, cyclists, and vehicles**, clusters foreground predictions with DBSCAN, and evaluates results on held-out synthetic scenes.

> **Scope:** This is NOT real LiDAR data, a RoboSense Airy 96 integration, Raspberry Pi/Jetson work, embedded deployment, ROS/ROS2, or real-world validation. It is also NOT PointNet. The project does not satisfy requirements for prior hands-on embedded hardware experience.

## Features
- Seeded synthetic point cloud generation and labels
- 9-dimensional per-point features combining normalized coordinates, reflectance and neighbour geometry
- PyTorch MLP classification with four semantic classes
- DBSCAN-derived approximate 3D boxes (not a trained object detector)
- Classification metrics, confusion matrix and Matplotlib visualization
- Unit tests for synthetic labels, feature shapes and empty detections

## Run with Windows PowerShell
Run in the repository root (where `requirements.txt` is located):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.perception --epochs 12 --train-scenes 30
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Start-Process ".\outputs\intersection_predictions.png"
```

This generates `outputs/intersection_predictions.png`, `outputs/metrics.json`, and `outputs/point_mlp.pt`. Output files are generated locally and do not need to be downloaded from this repository.

## Structure
- `src/perception.py` — simulation, feature extraction, neural network, training, evaluation, visualization, clustering
- `tests/test_perception.py` — three unit tests
- `requirements.txt` — dependencies

## Interpretation
Reported metrics refer **only to simulated point clouds**. The synthetic scene generator introduces simplified, separable object geometries. Performance on real sensors, varying weather, occlusion and domain shift is **unknown**.

## Interview description
“I built a small educational software prototype that simulates intersection point clouds, trains a PyTorch model for four point classes, and clusters foreground points into approximate 3D boxes. I validated it on separately seeded synthetic scenes; I have not connected a LiDAR or deployed on a Jetson.”

## Possible extensions (not implemented)
Public licensed LiDAR dataset; domain adaptation; sparse 3D architecture; ROS2 integration; field trials with real sensors; edge deployment.

## Credits
Educational example with synthetically generated data. No external point cloud dataset used.
