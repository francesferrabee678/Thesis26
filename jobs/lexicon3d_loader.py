import os
import numpy as np
from collections import OrderedDict

class Lexicon3DFeatureLoader:
    def __init__(self, feat_root, scan_dir, prefix='ptv3', max_cached_scenes=12):
        local = f"/tmp/{prefix}_features_matched"
        nfs = os.path.join(feat_root, f'{prefix}_features_matched')
        self.matched_dir = local if os.path.isdir(local) and len(os.listdir(local)) > 100 else nfs
        self.max_cached_scenes = max_cached_scenes
        self._cache = OrderedDict()

    def get_features(self, scene_id, target_xyz=None):
        if scene_id in self._cache:
            self._cache.move_to_end(scene_id)
            return self._cache[scene_id]
        path = os.path.join(self.matched_dir, f'{scene_id}.npy')
        with open(path, 'rb') as f:
            while f.read(64 << 20):
                pass
        arr = np.load(path, mmap_mode='r')
        self._cache[scene_id] = arr
        while len(self._cache) > self.max_cached_scenes:
            self._cache.popitem(last=False)
        return arr