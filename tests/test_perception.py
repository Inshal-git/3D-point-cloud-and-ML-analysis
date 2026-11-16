import unittest
import numpy as np
from src.perception import synthetic_scene, features, detect_instances, PointMLP

class TestPerception(unittest.TestCase):
    def test_scene_labels_and_shapes(self):
        pc, y, instance = synthetic_scene(15)
        self.assertEqual(pc.shape[1], 4)
        self.assertEqual(len(pc), len(y))
        self.assertEqual(len(y), len(instance))
        self.assertEqual(set(np.unique(y)), {0, 1, 2, 3})
        self.assertEqual(int((instance >= 0).sum()), len(y) - 650)

    def test_features_and_model(self):
        pc, _, _ = synthetic_scene(17)
        feat = features(pc)
        self.assertEqual(feat.shape, (len(pc), 9))
        self.assertEqual(PointMLP()(__import__('torch').from_numpy(feat[:10])).shape, (10, 4))

    def test_detector_empty_foreground(self):
        pc, _, _ = synthetic_scene(2)
        self.assertEqual(detect_instances(pc, np.zeros(len(pc), dtype=np.int64)), [])

if __name__ == '__main__':
    unittest.main()
