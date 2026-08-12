import itertools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/fferrabe/Thesis26/data/openscene_format/scannet_3d")
LEXICON = ROOT / "lexicon3d"
MODELS = ["concerto", "dinov2", "dinov3", "ptv3ppt", "uniscene3d"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PTS = 10000
SEED = 0


def load_features(model, scene):
    data = torch.load(LEXICON / model / f"{model}_features" / f"{scene}.pt", map_location="cpu", weights_only=False)
    feat = data["feat"].float().numpy()
    mask = data["mask_full"].numpy()
    full = np.zeros((mask.shape[0], feat.shape[1]), dtype=np.float32)
    full[mask] = feat
    return full


def valid_points(*feats):
    valid = np.ones(feats[0].shape[0], dtype=bool)
    for f in feats:
        valid &= np.isfinite(f).all(1) & (np.abs(f).sum(1) > 1e-8)
    return valid


class CKA:
    def __init__(self, models, dims):
        self.models = models
        self.n = 0
        self.sum = {m: torch.zeros(dims[m], device=DEVICE, dtype=torch.float64) for m in models}
        self.gram = {m: torch.zeros(dims[m], dims[m], device=DEVICE, dtype=torch.float64) for m in models}
        self.cross = {
            (a, b): torch.zeros(dims[a], dims[b], device=DEVICE, dtype=torch.float64)
            for a, b in itertools.combinations(models, 2)
        }

    @torch.no_grad()
    def update(self, feats):
        n = next(iter(feats.values())).shape[0]
        self.n += n
        for m in self.models:
            self.sum[m] += feats[m].sum(0)
            self.gram[m] += feats[m].T @ feats[m]
        for a, b in self.cross:
            self.cross[(a, b)] += feats[a].T @ feats[b]

    @torch.no_grad()
    def compute(self):
        n = self.n
        centered = {m: self.gram[m] - torch.outer(self.sum[m], self.sum[m]) / n for m in self.models}
        norm = {m: torch.linalg.norm(centered[m], "fro") for m in self.models}

        matrix = np.eye(len(self.models))
        scores = {}
        for a, b in self.cross:
            cross = self.cross[(a, b)] - torch.outer(self.sum[a], self.sum[b]) / n
            score = (torch.linalg.norm(cross, "fro") ** 2 / (norm[a] * norm[b] + 1e-12)).item()
            i, j = self.models.index(a), self.models.index(b)
            matrix[i, j] = matrix[j, i] = score
            scores[f"{a}_vs_{b}"] = score
        return matrix, scores


def run(scenes):
    rng = np.random.default_rng(SEED)
    acc = None

    for i, scene in enumerate(scenes):
        print(f"[{i + 1}/{len(scenes)}] {scene}")
        feats = {m: load_features(m, scene) for m in MODELS}
        idx = np.flatnonzero(valid_points(*feats.values()))
        if idx.size == 0:
            continue
        if idx.size > MAX_PTS:
            idx = rng.choice(idx, MAX_PTS, replace=False)

        x = {m: torch.as_tensor(feats[m][idx], device=DEVICE, dtype=torch.float64) for m in MODELS}
        if acc is None:
            acc = CKA(MODELS, {m: x[m].shape[1] for m in MODELS})
        acc.update(x)

    return acc.compute()


def print_matrix(matrix, models):
    print("\n=== Linear CKA ===")
    print("            " + "".join(f"{m[:11]:>13}" for m in models))
    for i, m in enumerate(models):
        print(f"{m[:11]:>11} " + "".join(f"{matrix[i, j]:13.4f}" for j in range(len(models))))


if __name__ == "__main__":
    scenes = sorted(p.stem for p in (LEXICON / MODELS[0] / f"{MODELS[0]}_features").glob("*.pt"))
    print(f"found {len(scenes)} scenes")

    matrix, scores = run(scenes)
    print_matrix(matrix, MODELS)

    with open("cka_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    np.save("cka_matrix.npy", matrix)
    print("\nsaved -> cka_scores.json, cka_matrix.npy")