import os
import glob
import json
import itertools

import numpy as np
import torch


ROOT = "/home/fferrabe/Thesis26/lexicon3d/evals/M3DRef-CLIP/output/ScanRefer"

MODELS = {
    "DINOv2":     f"{ROOT}/dinov2/inference/val/scores",
    "DINOv3":     f"{ROOT}/dinov3/inference/val/scores",
    "PTv3+PPT":    f"{ROOT}/ptv3ppt/inference/val/scores",
    "concerto":   f"{ROOT}/concerto/inference/val/scores",
    "Uniscene3D": f"{ROOT}/uniscene3d/inference/val/scores",
}

SCENE_DIR = "/extra/fferrabe/Thesis26/lexicon3d/evals/M3DRef-CLIP/dataset/scannetv2/val"

SCANREFER_META = "/extra/fferrabe/Thesis26/data/scanRef/ScanRefer_filtered_val.json"

GT_BOX_FIELD = "aabb_corner_xyz"
GT_ID_FIELD = "aabb_obj_ids"

IOU_THRESH = 0.25


def mm(box):
    """Convert box to (min,max)."""
    b = np.squeeze(np.asarray(box, dtype=np.float64))
    if b.ndim == 2:
        return b.min(0), b.max(0)
    return b[:3], b[3:]


def iou(a, b):
    amin, amax = mm(a)
    bmin, bmax = mm(b)

    inter = np.prod(np.clip(np.minimum(amax, bmax) - np.maximum(amin, bmin), 0, None))
    va = np.prod(np.clip(amax - amin, 0, None))
    vb = np.prod(np.clip(bmax - bmin, 0, None))
    union = va + vb - inter

    return inter / union if union > 0 else 0.0


def box_key(box):
    lo, hi = mm(box)
    return tuple(np.round(np.concatenate([lo, hi]), 3))


def load_models():
    """Load all score files."""
    models = {}

    for name, folder in MODELS.items():
        samples = {}

        for file in sorted(glob.glob(os.path.join(folder, "*.npz"))):
            scene = os.path.splitext(os.path.basename(file))[0]

            data = np.load(file)

            for i in range(len(data["object_ids"])):
                key = (
                    scene,
                    int(data["object_ids"][i]),
                    int(data["ann_ids"][i]),
                )

                samples[key] = (
                    data["scores"][i],   # logits
                    data["bounds"][i],   # proposal boxes
                )

        models[name] = samples
        print(f"{name:12s}: {len(samples)} samples")

    return models


def load_eval_types():
    eval_type = {}

    for item in json.load(open(SCANREFER_META)):
        key = (
            item["scene_id"],
            int(item["object_id"]),
            int(item["ann_id"]),
        )
        eval_type[key] = item["eval_type"]

    return eval_type


def build_dataset(models, eval_type):
    """Build aligned proposal vectors for every shared sample."""

    names = list(models.keys())

    shared = sorted(set.intersection(*(set(models[n]) for n in names)))

    gt_cache = {}

    aligned_logits = []
    aligned_ious = []
    unique = []

    for sample in shared:

        scene, object_id, _ = sample

        # load scene once
        if scene not in gt_cache:

            scene_file = os.path.join(SCENE_DIR, scene + ".pth")

            if not os.path.exists(scene_file):
                gt_cache[scene] = None
                continue

            gt = torch.load(scene_file, weights_only=False)

            ids = np.asarray(gt[GT_ID_FIELD]).astype(int)
            boxes = np.asarray(gt[GT_BOX_FIELD])

            gt_cache[scene] = {
                int(i): boxes[k]
                for k, i in enumerate(ids)
            }

        gt = gt_cache[scene]

        if gt is None:
            continue

        if object_id not in gt:
            continue

        if sample not in eval_type:
            continue

        canonical = {}
        boxes = []

        for model in names:
            _, proposals = models[model][sample]

            for box in proposals:
                key = box_key(box)

                if key not in canonical:
                    canonical[key] = len(boxes)
                    boxes.append(box)

        logits = {}

        for model in names:

            scores, proposals = models[model][sample]

            vec = np.full(len(boxes), -1e9)

            for score, box in zip(scores, proposals):
                vec[canonical[box_key(box)]] = score

            logits[model] = vec

        aligned_logits.append(logits)

        aligned_ious.append(
            np.array([iou(box, gt[object_id]) for box in boxes])
        )

        unique.append(eval_type[sample] == "unique")

    return aligned_logits, aligned_ious, np.asarray(unique)


def evaluate(subset, logits, ious, unique):
    correct = np.zeros(len(logits), dtype=bool)

    for i in range(len(logits)):

        total = None

        for model in subset:
            if total is None:
                total = logits[i][model].copy()
            else:
                total += logits[i][model]

        pred = np.argmax(total)

        correct[i] = ious[i][pred] >= IOU_THRESH

    return (
        correct[unique].mean() * 100,
        correct[~unique].mean() * 100,
        correct.mean() * 100,
    )


def main():

    models = load_models()
    eval_type = load_eval_types()

    logits, ious, unique = build_dataset(models, eval_type)

    names = list(models.keys())

    print(f"\nShared samples: {len(logits)}")
    print(f"Unique:   {unique.sum()}")
    print(f"Multiple: {(~unique).sum()}")

    print()
    print(f"{'Combination':45s} {'Unique':>8} {'Multiple':>10} {'Overall':>8}")
    print("-" * 75)

    # single models
    for model in names:
        u, m, o = evaluate([model], logits, ious, unique)

        print(
            f"{model:45s}"
            f"{u:8.2f}"
            f"{m:10.2f}"
            f"{o:8.2f}"
        )

    # every ensemble
    for r in range(2, len(names) + 1):

        for combo in itertools.combinations(names, r):

            u, m, o = evaluate(combo, logits, ious, unique)

            print(
                f"{'+'.join(combo):45s}"
                f"{u:8.2f}"
                f"{m:10.2f}"
                f"{o:8.2f}"
            )


if __name__ == "__main__":
    main()