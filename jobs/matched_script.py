import os, glob, torch, numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

FEAT_ROOT = "/home/fferrabe/Thesis26/data/openscene_format/scannet_3d/lexicon3d/dinov3"
SCENE_DATA_ROOT = "/extra/fferrabe/Thesis26/lexicon3d/evals/M3DRef-CLIP/dataset/scannetv2"
OUT_DIR = os.path.join(FEAT_ROOT, "dinov3_features_matched")
os.makedirs(OUT_DIR, exist_ok=True)

for split in ["train", "val"]:
    split_dir = os.path.join(SCENE_DATA_ROOT, split)
    scene_paths = glob.glob(os.path.join(split_dir, "scene[0-9][0-9][0-9][0-9]_[0-9][0-9].pth"))
    print(f"{split}: {len(scene_paths)} scene files matched")

    for scene_path in tqdm(scene_paths, desc=f"Precomputing {split}"):
        scene_id = os.path.splitext(os.path.basename(scene_path))[0]
        out_path = os.path.join(OUT_DIR, f"{scene_id}.npy")
        if os.path.exists(out_path):
            continue

        scene_data = torch.load(scene_path, weights_only=False)
        if not isinstance(scene_data, dict) or "xyz" not in scene_data:
            print(f"  skipping {scene_id}: unexpected format ({type(scene_data)})")
            continue

        target_xyz = scene_data["xyz"].astype(np.float32)

        feat_path = os.path.join(FEAT_ROOT, "dinov3_features", f"{scene_id}.pt")
        pts_path = os.path.join(FEAT_ROOT, "dinov3_points", f"{scene_id}.npy")
        d = torch.load(feat_path, map_location="cpu")
        feat = d["feat"].numpy()  # already float16 on disk — no .float() upcast needed
        raw_locs = np.load(pts_path).astype(np.float32)

        tree = cKDTree(raw_locs)
        _, idx = tree.query(target_xyz, k=1, workers=-1)
        matched = feat[idx].astype(np.float16)

        np.save(out_path, matched)

print("Done.")