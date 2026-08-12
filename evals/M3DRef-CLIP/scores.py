"""
Capture per-candidate scores at inference WITHOUT modifying any source file. (v4)

v4 adds instrumentation for evaluation jitter:
  - dup    : annotations predicted more than once (chunk wrap-around padding)
  - flips  : of those, how many got a DIFFERENT box the second time
  - margin : top1 - top2 score gap, i.e. how close the decisions are
"""

import os
import hydra
import torch
import numpy as np
import lightning.pytorch as pl
from tqdm import tqdm

from m3drefclip.data.data_module import DataModule


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(cfg):
    pl.seed_everything(cfg.test_seed, workers=True)
    os.makedirs(os.path.join(cfg.pred_path, cfg.data.inference.split), exist_ok=True)

    data_module = DataModule(cfg.data)
    cfg.data.evaluator.verbose = True
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)

    assert cfg.ckpt_path is not None, "Error: Checkpoint path is not provided."
    assert os.path.exists(cfg.ckpt_path), f"Error: {cfg.ckpt_path} does not exist."

    cls = type(model)
    orig_parse = cls._parse_pred_results
    orig_save = cls._save_predictions
    state = {"reported": False, "checked": 0, "bad": 0,
             "seen": {}, "dup": 0, "flips": 0, "margins": []}
    print(f"[capture] patching {cls.__module__}.{cls.__name__} in memory")

    def parse_with_scores(self, data_dict, output_dict):
        results = orig_parse(self, data_dict, output_dict)

        batch_size, lang_chunk_size = data_dict["ann_id"].shape
        raw = output_dict["pred_aabb_scores"]

        ref_idx = raw.argmax(dim=1).reshape(batch_size, lang_chunk_size, -1)[..., 0]

        if raw.dim() == 3:
            scores = raw.permute(0, 2, 1).contiguous()
        elif raw.dim() == 2:
            scores = raw.reshape(batch_size, lang_chunk_size, -1)
        else:
            raise RuntimeError(f"unexpected pred_aabb_scores shape {tuple(raw.shape)}")

        if not torch.equal(scores.argmax(dim=2), ref_idx):
            raise RuntimeError(
                f"score extraction mismatch (shape {tuple(raw.shape)}, "
                f"batch={batch_size}, chunk={lang_chunk_size}).")

        if not state["reported"]:
            print(f"[capture] pred_aabb_scores {tuple(raw.shape)} -> "
                  f"{tuple(scores.shape)}  (argmax verified)")
            state["reported"] = True

        scores_np = scores.detach().float().cpu().numpy()

        for i in range(batch_size):
            bounds = (output_dict["pred_aabb_min_max_bounds"][i]
                      + data_dict["scene_center_xyz"][i]).detach().float().cpu().numpy()
            for j in range(lang_chunk_size):
                key = (data_dict["scene_id"][i],
                       data_dict["object_id"][i][j].item(),
                       data_dict["ann_id"][i][j].item())
                if key not in results:
                    continue
                results[key]["scores"] = scores_np[i, j]
                results[key]["proposal_bounds"] = bounds

        # ---- bounds/argmax consistency check (unchanged) ----
        for key, val in results.items():
            if "scores" not in val:
                continue
            idx = int(np.argmax(val["scores"]))
            got = val["proposal_bounds"][idx]
            want = np.squeeze(np.asarray(val["aabb_bound"]))
            if want.ndim == 2 and want.shape == (2, 3):
                state["checked"] += 1
                if not np.allclose(got, want, atol=1e-3):
                    state["bad"] += 1
                    if state["bad"] <= 3:
                        print(f"  [capture] MISMATCH {key}: argmax={idx}\n"
                              f"    bounds[argmax] = {got.tolist()}\n"
                              f"    original pred  = {want.tolist()}")
        if state["checked"] >= 200 and state["bad"] > 0.5 * state["checked"]:
            raise RuntimeError(
                f"{state['bad']}/{state['checked']} annotations mismatch -- more "
                f"than half, so the extraction is fundamentally wrong. Aborting.")

        # ---- JITTER INSTRUMENTATION ----
        for key, val in results.items():
            b = np.asarray(val["aabb_bound"])
            if b.size == 0:
                continue
            h = np.round(np.squeeze(b), 3).tobytes()
            if key in state["seen"]:
                state["dup"] += 1
                if state["seen"][key] != h:
                    state["flips"] += 1
            state["seen"][key] = h
            if "scores" in val:
                s = np.sort(np.asarray(val["scores"], dtype=np.float64))[::-1]
                if s.size >= 2:
                    state["margins"].append(float(s[0] - s[1]))

        return results

    def save_with_scores(self, predictions):
        orig_save(self, predictions)

        root = os.path.join(cfg.pred_path, cfg.data.inference.split, "scores")
        os.makedirs(root, exist_ok=True)

        per_scene = {}
        for key, value in predictions.items():
            if "scores" not in value:
                continue
            per_scene.setdefault(key[0], []).append(
                (key[1], key[2], value["scores"], value["proposal_bounds"]))

        for sid, rows in tqdm(per_scene.items(), desc="Saving scores"):
            np.savez_compressed(
                os.path.join(root, f"{sid}.npz"),
                object_ids=np.array([r[0] for r in rows], dtype=np.int32),
                ann_ids=np.array([r[1] for r in rows], dtype=np.int32),
                scores=np.stack([np.asarray(r[2], dtype=np.float32) for r in rows]),
                bounds=np.stack([np.asarray(r[3], dtype=np.float32) for r in rows]),
            )
        n_ann = sum(len(v) for v in per_scene.values())
        self.print(f"==> Scores: {len(per_scene)} scenes / {n_ann} annotations, "
                   f"verified {state['checked'] - state['bad']}/{state['checked']} "
                   f"against the original predictions")
        self.print(f"==> at {os.path.abspath(root)}")

    cls._parse_pred_results = parse_with_scores
    cls._save_predictions = save_with_scores

    orig_test_end = cls.on_test_epoch_end

    def test_end_with_eval(self):
        preds, gts = {}, {}
        for p, g in self.val_test_step_outputs:
            preds.update(p)
            gts.update(g)

        try:
            self.evaluator.set_ground_truths(gts)
            res = self.evaluator.evaluate(preds)
            self.print("=" * 56)
            self.print("OFFICIAL EVALUATOR (same code the validation loop uses)")
            self.print("=" * 56)
            for metric, bd in res.items():
                for k, v in bd.items():
                    self.print(f"  {metric}_{k}: "
                               f"{v:.4f}" if isinstance(v, float) else f"  {metric}_{k}: {v}")
        except Exception as e:
            self.print(f"[capture] official evaluation failed: {e}")

        # ---- JITTER REPORT ----
        n_uniq = len(state["seen"])
        mg = np.asarray(state["margins"]) if state["margins"] else np.zeros(1)
        self.print("=" * 56)
        self.print("EVAL JITTER")
        self.print("=" * 56)
        self.print(f"  unique annotations           : {n_uniq}")
        self.print(f"  duplicate predictions        : {state['dup']} "
                   f"({100*state['dup']/max(n_uniq,1):.1f}% of annotations)")
        self.print(f"  of those, box CHANGED        : {state['flips']}"
                   + (f" ({100*state['flips']/state['dup']:.1f}%)" if state['dup'] else ""))
        self.print(f"  -> arbitrary decisions       : ~{state['flips']} samples "
                   f"= {100*state['flips']/max(n_uniq,1):.2f} pp of accuracy")
        self.print(f"  top1-top2 margin: median {np.median(mg):.4f} | "
                   f"<0.01: {100*(mg<0.01).mean():.1f}% | <0.1: {100*(mg<0.1).mean():.1f}%")

        root = os.path.join(cfg.pred_path, cfg.data.inference.split)
        os.makedirs(root, exist_ok=True)
        keys = list(gts.keys())
        np.savez_compressed(
            os.path.join(root, "gt.npz"),
            keys=np.array([f"{k[0]}|{k[1]}|{k[2]}" for k in keys]),
            boxes=np.stack([np.asarray(gts[k]["aabb_bound"], dtype=np.float32)[0]
                            for k in keys]),
            eval_types=np.array([str(gts[k]["eval_type"]) for k in keys]),
        )
        self.print(f"==> GT for {len(keys)} annotations -> {os.path.join(root, 'gt.npz')}")
        orig_test_end(self)

    cls.on_test_epoch_end = test_end_with_eval

    trainer = pl.Trainer(accelerator=cfg.trainer.accelerator, devices=1,
                         max_epochs=1, logger=False)
    try:
        trainer.test(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)
    finally:
        cls._parse_pred_results = orig_parse
        cls._save_predictions = orig_save
        cls.on_test_epoch_end = orig_test_end


if __name__ == '__main__':
    main()