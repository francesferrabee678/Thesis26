import os
import torch
import numpy as np
from collections import OrderedDict
from scipy.spatial import cKDTree


def _load_axis_align_matrix(scan_dir, scene_id):
    meta_path = os.path.join(scan_dir, scene_id, scene_id + '.txt')
    axis_align_matrix = np.eye(4, dtype=np.float32)
    with open(meta_path, 'r') as f:
        for line in f:
            if 'axisAlignment' in line:
                vals = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                axis_align_matrix = np.array(vals, dtype=np.float32).reshape((4, 4))
                break
    return axis_align_matrix


class Lexicon3DFeatureLoader:
    def __init__(self, feat_root, scan_dir, prefix='dinov2', max_cached_scenes=6):
        self.feat_root = feat_root
        self.scan_dir = scan_dir
        self.prefix = prefix
        self.max_cached_scenes = max_cached_scenes
        self._cache = OrderedDict()

    def _load_scene(self, scene_id):
        p = self.prefix
        feat_path = os.path.join(self.feat_root, f'{p}_features', f'{scene_id}.pt')
        pts_path = os.path.join(self.feat_root, f'{p}_points', f'{scene_id}.npy')

        d = torch.load(feat_path, map_location='cpu')
        feat = d['feat'].float().numpy()
        raw_locs = np.load(pts_path).astype(np.float32)

        if feat.shape[0] != raw_locs.shape[0]:
            raise ValueError(f"[{scene_id}] feat {feat.shape[0]} != points {raw_locs.shape[0]}")

        axis_align_matrix = _load_axis_align_matrix(self.scan_dir, scene_id)
        locs_4 = np.concatenate([raw_locs, np.ones((raw_locs.shape[0], 1), dtype=np.float32)], axis=1)
        locs_aligned = (locs_4 @ axis_align_matrix.T)[:, :3]

        return locs_aligned, feat, cKDTree(locs_aligned)

    def get_features(self, scene_id, target_xyz):
        if scene_id in self._cache:
            self._cache.move_to_end(scene_id)
        else:
            self._cache[scene_id] = self._load_scene(scene_id)
            if len(self._cache) > self.max_cached_scenes:
                self._cache.popitem(last=False)

        _, feat, tree = self._cache[scene_id]
        _, idx = tree.query(target_xyz, k=1, workers=-1)
        return feat[idx]