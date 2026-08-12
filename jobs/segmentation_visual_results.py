#!/usr/bin/env python3
"""
Render Lexicon3D semantic-segmentation results, one image per scene — same
spirit as the repo's visualize_feature.py, but colored by predicted class
instead of PCA features. No grid assembly; arrange the PNGs however you like.

You give it scenes and a model name:

  # a probe model (dinov2 / lseg / clip / sd / vjepa / svd)
  python visualize_segmentation.py --data_dir dataset/ScanNet/openscene \
      --feat_root dataset/lexicon3d --model dinov2 --ckpt ckpts/dinov2.ckpt \
      --scenes scene0011_00 scene0435_00

  # the ground truth
  python visualize_segmentation.py --data_dir dataset/ScanNet/openscene \
      --model gt --scenes scene0011_00

  # the plain RGB point cloud
  python visualize_segmentation.py --data_dir dataset/ScanNet/openscene \
      --model rgb --scenes scene0011_00

  # externally dumped predictions, e.g. Swin3D ({pred_dir}/{scene}.npy, (N,) ints)
  python visualize_segmentation.py --data_dir dataset/ScanNet/openscene \
      --model swin3d --pred_dir preds/swin3d --scenes scene0011_00

Writes {out}/{scene}_{model}.png.

Data layout (Lexicon3D / OpenScene):
  {data_dir}/scannet_3d/{split}/{scene}_vh_clean_2.pth      -> (coords, colors, labels)
  {data_dir}/process_ply/label_20/{scene}.pth               -> labels used by the probes (-1 ignore)
  {feat_root}/{model}/{model}_features/{scene}.pt           -> {'feat': (M,C), 'mask_full': (N,) bool}

The probe checkpoint is a Lightning .ckpt from evals/segmentation/main.py
(the probe is a single nn.Linear(C, 20) stored as 'model.fc.*').

Camera matches visualize_feature.py: pitch 45, yaw 0 (auto 180 for scene0031_*),
distance 0.8 x bbox diagonal, fov 60, unlit points, white background. Uses
Open3D offscreen if available, otherwise a pure-numpy fallback.
"""

import argparse
import os
from os.path import join, exists

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def torch_load(path):
    """torch.load across PyTorch versions: 2.6+ defaults weights_only=True,
    which breaks Lightning ckpts and numpy-containing scene .pth files."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        return torch.load(path, map_location='cpu')


# --------------------------------------------------------------------------
# ScanNet 20-class benchmark palette. Index = train id 0..19; -1/255 = black.
# --------------------------------------------------------------------------
SCANNET20_PALETTE = np.array([
    (174, 199, 232),  # wall
    (152, 223, 138),  # floor
    (31, 119, 180),   # cabinet
    (255, 187, 120),  # bed
    (188, 189, 34),   # chair
    (140, 86, 75),    # sofa
    (255, 152, 150),  # table
    (214, 39, 40),    # door
    (197, 176, 213),  # window
    (148, 103, 189),  # bookshelf
    (196, 156, 148),  # picture
    (23, 190, 207),   # counter
    (247, 182, 210),  # desk
    (219, 219, 141),  # curtain
    (255, 127, 14),   # refrigerator
    (158, 218, 229),  # shower curtain
    (44, 160, 44),    # toilet
    (112, 128, 144),  # sink
    (227, 119, 194),  # bathtub
    (82, 84, 163),    # otherfurniture
], dtype=np.uint8)


def labels_to_colors(labels):
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    out = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    valid = (labels >= 0) & (labels < len(SCANNET20_PALETTE))
    out[valid] = SCANNET20_PALETTE[labels[valid]]
    return out


# --------------------------------------------------------------------------
# Data loading / prediction
# --------------------------------------------------------------------------
def load_scene(data_dir, split, scene):
    """coords (N,3) float64, rgb (N,3) uint8, gt labels (N,) int (-1 ignore)."""
    if torch is None:
        raise RuntimeError('PyTorch is required (pip install torch).')
    pth = join(data_dir, 'scannet_3d', split, scene + '_vh_clean_2.pth')
    if not exists(pth):
        pth = join(data_dir, 'scannet_3d', split, scene + '.pth')
    data = torch_load(pth)
    coords, colors, labels = data[0], data[1], data[2]
    coords = np.asarray(coords, dtype=np.float64)

    colors = np.asarray(colors)
    if colors.dtype != np.uint8:  # OpenScene stores colors in [-1, 1]
        if colors.min() < -0.01:
            colors = (colors + 1.0) * 127.5
        elif colors.max() <= 1.001:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    label20 = join(data_dir, 'process_ply', 'label_20', scene + '.pth')
    if exists(label20):
        labels = torch_load(label20)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    labels[labels == 255] = -1
    return coords, colors, labels


def load_probe_weights(ckpt_path):
    ckpt = torch_load(ckpt_path)
    state = ckpt.get('state_dict', ckpt)
    wkey = next((k for k in state if k.endswith('fc.weight')), None)
    bkey = next((k for k in state if k.endswith('fc.bias')), None)
    if wkey is None or bkey is None:
        raise KeyError(f'No fc.weight/fc.bias in {ckpt_path}; keys: {list(state)[:10]}')
    return state[wkey].float(), state[bkey].float()


def predict_scene(feat_root, model, scene, W, b, feat_dir=None):
    """(N,) predictions in [0,19]; -1 where the point has no fused feature."""
    fdir = feat_dir or join(feat_root, model, f'{model}_features')
    d = torch_load(join(fdir, scene + '.pt'))
    feat, mask = d['feat'].float(), np.asarray(d['mask_full']).astype(bool)
    if feat.shape[1] != W.shape[1]:
        raise ValueError(f'{model}/{scene}: feat dim {feat.shape[1]} != '
                         f'probe dim {W.shape[1]} — wrong checkpoint?')
    with torch.no_grad():
        pred = (feat @ W.t() + b).argmax(dim=1).numpy()
    full = np.full(mask.shape[0], -1, dtype=np.int64)
    full[mask] = pred
    return full


def nn_fill(coords, labels):
    """1-NN propagate labels to points with -1 (dense render like the paper)."""
    labels = labels.copy()
    known = labels >= 0
    if known.all() or not known.any():
        return labels
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print('[warn] scipy missing — unpredicted points stay black.')
        return labels
    tree = cKDTree(coords[known])
    _, idx = tree.query(coords[~known], k=1, workers=-1)
    labels[~known] = labels[known][idx]
    return labels


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
class O3DRenderer:
    def __init__(self, res=1024, point_size=4.0):
        import open3d as o3d
        from open3d.visualization import rendering
        self.o3d = o3d
        self.render = rendering.OffscreenRenderer(res, res)
        try:
            self.render.scene.view.set_post_processing(False)
        except Exception:
            pass
        self.render.scene.set_background([1, 1, 1, 1])
        self.mtl = rendering.MaterialRecord()
        self.mtl.shader = 'defaultUnlit'
        self.mtl.point_size = float(point_size)

    def __call__(self, coords, colors_uint8, pitch=45.0, yaw=0.0):
        o3d = self.o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(colors_uint8.astype(np.float64) / 255.0)
        self.render.scene.clear_geometry()
        self.render.scene.add_geometry('pcd', pcd, self.mtl)
        center = pcd.get_center()
        cam_dist = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound()) * 0.8
        p, y = np.deg2rad(pitch), np.deg2rad(yaw)
        off = cam_dist * np.array([np.cos(p) * np.cos(y),
                                   np.cos(p) * np.sin(y),
                                   np.sin(p)])
        self.render.setup_camera(60.0, center, center + off, [0, 0, 1])
        return np.asarray(self.render.render_to_image())[:, :, :3].astype(np.uint8)


def render_numpy(coords, colors_uint8, pitch=45.0, yaw=0.0, res=1024,
                 point_px=2, supersample=2, fov_deg=60.0):
    center = coords.mean(0)
    diag = np.linalg.norm(coords.max(0) - coords.min(0))
    p, y = np.deg2rad(pitch), np.deg2rad(yaw)
    cam = center + 0.8 * diag * np.array([np.cos(p) * np.cos(y),
                                          np.cos(p) * np.sin(y),
                                          np.sin(p)])
    fwd = center - cam
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)

    rel = coords - cam
    zc = rel @ fwd
    keep = zc > 1e-6
    rel, zc, cols = rel[keep], zc[keep], colors_uint8[keep]
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    xs = (rel @ right) / zc * f
    ys = (rel @ upv) / zc * f

    R = res * supersample
    half = R / 2.0
    scale = half * 0.95 / max(np.abs(xs).max(), np.abs(ys).max(), 1e-9)
    px = (half + xs * scale).astype(np.int64)
    py = (half - ys * scale).astype(np.int64)

    order = np.argsort(-zc)  # far -> near, so near points overwrite
    px, py, cols = px[order], py[order], cols[order]

    img = np.full((R, R, 3), 255, dtype=np.uint8)
    r = max(1, int(round(point_px * supersample / 2)))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r + 1:
                continue
            img[np.clip(py + dy, 0, R - 1), np.clip(px + dx, 0, R - 1)] = cols
    if supersample > 1:
        img = img.reshape(res, supersample, res, supersample, 3).mean((1, 3)).astype(np.uint8)
    return img


def get_renderer(backend, res, point_size):
    if backend in ('auto', 'open3d'):
        try:
            r = O3DRenderer(res=res, point_size=point_size)
            print('[info] rendering with Open3D OffscreenRenderer')
            return r
        except Exception as e:
            if backend == 'open3d':
                raise
            print(f'[warn] Open3D offscreen unavailable ({e}); using numpy '
                  f'fallback. (xvfb-run also fixes headless Open3D.)')
    print('[info] rendering with numpy fallback')
    return lambda c, col, pitch=45.0, yaw=0.0: render_numpy(
        c, col, pitch, yaw, res=res, point_px=int(round(point_size / 2)) + 1)


def autocrop(img, pad=8, bg=255):
    mask = (img < bg - 2).any(-1)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img[max(ys.min() - pad, 0):ys.max() + pad + 1,
               max(xs.min() - pad, 0):xs.max() + pad + 1]


def save_png(img, path):
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, img)
    except ImportError:
        from PIL import Image
        Image.fromarray(img).save(path)
    print(f'[done] {path}')


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', required=True,
                    help='OpenScene-preprocessed ScanNet root (scannet_3d/, process_ply/)')
    ap.add_argument('--scenes', nargs='+', required=True, help='e.g. scene0011_00')
    ap.add_argument('--model', required=True,
                    help="'gt', 'rgb', a fused-feature model (dinov2/lseg/clip/sd/"
                         "vjepa/svd, needs --ckpt), or any name with --pred_dir")
    ap.add_argument('--ckpt', default=None, help='linear-probe .ckpt for --model')
    ap.add_argument('--feat_root', default=None,
                    help='root of fused features: {feat_root}/{model}/{model}_features/')
    ap.add_argument('--feat_dir', default=None, help='override the features dir directly')
    ap.add_argument('--pred_dir', default=None,
                    help='dir with precomputed {scene}.npy full-length predictions')
    ap.add_argument('--split', default='val')
    ap.add_argument('--pitch', type=float, default=45.0)
    ap.add_argument('--yaw', type=float, default=None,
                    help='default 0 (180 for scene0031_*, like visualize_feature.py)')
    ap.add_argument('--no_fill', action='store_true',
                    help='leave points without features black instead of 1-NN filling')
    ap.add_argument('--no_crop', action='store_true')
    ap.add_argument('--backend', choices=['auto', 'open3d', 'numpy'], default='auto')
    ap.add_argument('--res', type=int, default=1024)
    ap.add_argument('--point_size', type=float, default=4.0)
    ap.add_argument('--out', default='segmentation_vis')
    args = ap.parse_args()

    model = args.model.lower()
    is_probe = model not in ('gt', 'rgb') and args.pred_dir is None
    if is_probe:
        if args.ckpt is None:
            raise SystemExit(f"--model {args.model} needs --ckpt (or --pred_dir).")
        if args.feat_root is None and args.feat_dir is None:
            raise SystemExit('--feat_root (or --feat_dir) is required for probe models.')
        W, b = load_probe_weights(args.ckpt)

    os.makedirs(args.out, exist_ok=True)
    renderer = get_renderer(args.backend, args.res, args.point_size)

    for scene in args.scenes:
        coords, rgb, gt = load_scene(args.data_dir, args.split, scene)

        if model == 'rgb':
            colors = rgb
        elif model == 'gt':
            colors = labels_to_colors(gt)
        else:
            if args.pred_dir is not None:
                pred = np.load(join(args.pred_dir, scene + '.npy')).astype(np.int64).reshape(-1)
                if pred.shape[0] != coords.shape[0]:
                    raise ValueError(f'{scene}: pred length {pred.shape[0]} != '
                                     f'point count {coords.shape[0]}')
            else:
                pred = predict_scene(args.feat_root, model, scene, W, b,
                                     feat_dir=args.feat_dir)
            if not args.no_fill:
                pred = nn_fill(coords, pred)
            colors = labels_to_colors(pred)

        yaw = args.yaw
        if yaw is None:
            yaw = 180.0 if scene.startswith('scene0031_') else 0.0
            yaw = 270.0 if scene.startswith('scene0030_') else 0.0



        img = renderer(coords, colors, pitch=args.pitch, yaw=yaw)
        if not args.no_crop:
            img = autocrop(img)
        save_png(img, join(args.out, f'{scene}_{args.model}.png'))


if __name__ == '__main__':
    main()