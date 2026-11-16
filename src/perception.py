"""Synthetic intersection LiDAR data, semantic segmentation and object extraction.

Educational CPU simulation only; no real LiDAR measurement or embedded deployment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn

CLASSES = ["ground", "pedestrian", "cyclist", "vehicle"]
COLORS = ["#818b99", "#e7a141", "#35a6a1", "#ce5e73"]


def synthetic_scene(seed: int, n_ground: int = 650):
    """Return XYZI float32 points, point class labels and instance labels."""
    rng = np.random.default_rng(seed)
    pts, labels, ids = [], [], []
    xy = rng.uniform(-14, 14, size=(n_ground, 2))
    ground = np.column_stack((xy, rng.normal(0, .025, n_ground),
                              rng.uniform(.1, .25, n_ground)))
    pts.append(ground)
    labels.extend([0] * n_ground)
    ids.extend([-1] * n_ground)
    instance = 0
    occupied = []
    for cls, n in [(1, 3), (2, 2), (3, 3)]:
        for _ in range(n):
            for trial in range(200):
                x, y = rng.uniform(-11, 11, 2)
                if all(np.hypot(x - a, y - b) > 4.0 for a, b in occupied):
                    break
            occupied.append((x, y))
            k = {1: 80, 2: 95, 3: 180}[cls]
            if cls == 1:
                local = np.column_stack((rng.normal(0, .18, k),
                                         rng.normal(0, .18, k), rng.uniform(.2, 1.82, k)))
                strength = rng.uniform(.4, .7, k)
            elif cls == 2:
                local = np.column_stack((rng.normal(0, .62, k),
                                         rng.normal(0, .16, k), rng.uniform(.25, 1.6, k)))
                strength = rng.uniform(.28, .55, k)
            else:
                local = np.column_stack((rng.uniform(-1.8, 1.8, k),
                                         rng.uniform(-.82, .82, k), rng.uniform(.3, 1.55, k)))
                strength = rng.uniform(.7, .95, k)
            a = rng.uniform(0, 2 * np.pi)
            xyrot = local[:, :2] @ np.array([[np.cos(a), np.sin(a)],
                                              [-np.sin(a), np.cos(a)]])
            obj = np.column_stack((xyrot[:, 0] + x, xyrot[:, 1] + y,
                                   local[:, 2], strength))
            pts.append(obj)
            labels.extend([cls] * k)
            ids.extend([instance] * k)
            instance += 1
    return (np.vstack(pts).astype("float32"),
            np.asarray(labels, dtype=np.int64),
            np.asarray(ids, dtype=np.int64))


def features(points: np.ndarray, neighbours: int = 16) -> np.ndarray:
    """Normalize XYZI and calculate local nearest-neighbour geometry."""
    xyz = points[:, :3]
    idx = cKDTree(xyz).query(xyz, k=min(neighbours, len(xyz)))[1]
    if idx.ndim == 1:
        idx = idx[:, None]
    local = xyz[idx]
    spread = local.std(axis=1)
    height = local[:, :, 2].max(axis=1) - local[:, :, 2].min(axis=1)
    nearest = (np.linalg.norm(local[:, 1, :] - xyz, axis=1)
               if idx.shape[1] > 1 else np.zeros(len(xyz)))
    normalized = np.column_stack((xyz[:, :2] / 14, xyz[:, 2] / 2, points[:, 3]))
    return np.column_stack((normalized, spread, height, nearest)).astype("float32")


def data_split(start: int, scenes: int):
    xs, ys = [], []
    for seed in range(start, start + scenes):
        pc, y, _ = synthetic_scene(seed)
        xs.append(features(pc))
        ys.append(y)
    return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))


class PointMLP(nn.Module):
    """Simple pointwise MLP; NOT PointNet."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)


def detect_instances(points: np.ndarray, predictions: np.ndarray):
    """DBSCAN on predicted foreground; approximate axis-aligned 3D boxes."""
    mask = predictions != 0
    if mask.sum() == 0:
        return []
    assignments = DBSCAN(eps=.72, min_samples=10).fit_predict(points[mask, :3])
    detections = []
    for i in np.unique(assignments):
        if i < 0:
            continue
        indices = np.flatnonzero(mask)[assignments == i]
        chunk = points[indices, :3]
        if len(chunk) < 15:
            continue
        majority = np.bincount(predictions[indices], minlength=4).argmax()
        detections.append({
            "class": CLASSES[int(majority)],
            "points": int(len(indices)),
            "min_xyz": np.round(chunk.min(axis=0), 2).tolist(),
            "max_xyz": np.round(chunk.max(axis=0), 2).tolist(),
        })
    return detections


def figure(points, truth, prediction, out: Path):
    fig = plt.figure(figsize=(14, 6), layout="constrained")
    for j, (labels, title) in enumerate([
        (truth, "Synthetic ground truth"), (prediction, "Model prediction")
    ]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        for i, name in enumerate(CLASSES):
            p = points[labels == i]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                       s=2 if i == 0 else 5, alpha=.7,
                       color=COLORS[i], label=name)
        ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title=title)
        ax.set_xlim(-14, 14)
        ax.set_ylim(-14, 14)
        ax.set_zlim(0, 3)
        ax.view_init(elev=25, azim=-65)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Intersection perception · educational simulated point cloud", fontsize=14)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def run(out_dir: Path, epochs: int, train_scenes: int):
    torch.manual_seed(7)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = data_split(100, train_scenes)
    x_test, y_test = data_split(1000, 6)
    model = PointMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    counts = torch.bincount(y_train, minlength=4).float()
    weights = (counts.sum() / counts).sqrt()
    weights /= weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    history = []
    for epoch in range(epochs):
        model.train()
        batches = torch.randperm(len(x_train)).split(1024)
        total_loss = 0
        for b in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[b]), y_train[b])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            pred = model(x_test).argmax(axis=1)
        acc = (pred == y_test).float().mean().item()
        history.append({
            "epoch": epoch + 1,
            "loss": round(total_loss / len(x_train), 5),
            "test_accuracy": round(acc, 4),
        })
        print(f'Epoch {epoch + 1:02d}: train_loss={history[-1]["loss"]:.4f}, synthetic_test_accuracy={acc:.3f}')
    with torch.no_grad():
        prediction = model(x_test).argmax(axis=1).numpy()
    truth = y_test.numpy()
    report = classification_report(truth, prediction,
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, prediction, labels=range(4)).tolist()
    pointcloud, scene_truth, scene_ids = synthetic_scene(2042)
    with torch.no_grad():
        scene_pred = model(torch.from_numpy(features(pointcloud))).argmax(axis=1).numpy()
    detections = detect_instances(pointcloud, scene_pred)
    figure(pointcloud, scene_truth, scene_pred, out_dir / "intersection_predictions.png")
    torch.save(model.state_dict(), out_dir / "point_mlp.pt")
    result = {
        "warning": "All data are synthetic. No real LiDAR, Jetson, Raspberry Pi, ROS or hardware testing.",
        "seed": 7, "train_scenes": train_scenes, "heldout_scenes": 6,
        "metrics": report, "confusion_matrix": matrix,
        "demo_scene_gt_instances": int(len(np.unique(scene_ids[scene_ids >= 0]))),
        "demo_scene_predicted_clusters": len(detections),
        "detections": detections, "training_history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved outputs in {out_dir}: intersection_predictions.png, metrics.json, point_mlp.pt")
    print("Macro F1 on held-out synthetic scenes:", round(report["macro avg"]["f1-score"], 3))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-scenes", type=int, default=30)
    args = parser.parse_args()
    run(args.out, args.epochs, args.train_scenes)


if __name__ == "__main__":
    main()
