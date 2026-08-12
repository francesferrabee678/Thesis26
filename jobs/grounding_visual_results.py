#!/usr/bin/env python3
"""
Visualize 3D visual grounding results from the Lexicon3D / M3DRef-CLIP eval.

For each language query it renders the RGB point cloud with:
  - the ground-truth box of the referred object (green)
  - the box each model *chose* for that query (one color per model)
  - optionally the other GT object boxes as thin gray context
    (--show_all_objects). With use_gt_proposal=True this gray set is exactly
    the candidate pool the model picks from. Wall/floor/ceiling instance
    boxes (which span the whole room) are filtered out unless
    --keep_structural is given.
and writes one PNG per query with the query text + per-model result printed
in a caption strip below the image.

GT-proposal mode: when the model selects among ground-truth boxes
(detector.use_gt_proposal=True, the Lexicon3D setting), a correct pick
coincides exactly with the GT box (IoU 1.0) and a wrong pick usually has
IoU ~0 — the metric is effectively accuracy. Correct picks are therefore
drawn as a slightly inflated concentric outline around the green GT box
(so both stay visible) and labeled "correct" instead of an IoU number.

It consumes the files the M3DRef-CLIP pipeline already produces:

  scene data   {scene_data_dir}/{scene}.pth
               (preprocess_all_data.py output: dict with 'xyz', 'rgb',
                'aabb_corner_xyz', 'aabb_obj_ids', ...; coords are axis-aligned)
  predictions  {pred_dir}/{scene}.json
               (written by test.py at {pred_path}/{split}/:
                [{"object_id": int, "ann_id": int, "aabb": [n][8][3]}, ...])
  language     ScanRefer_filtered_val.json (or Nr3D) for the descriptions

Examples:

  # one model
  python grounding_visual_results.py \
      --scene_data_dir data/scannetv2/val \
      --lang_json data/scanrefer/ScanRefer_filtered_val.json \
      --preds ptv3ppt=output/ScanRefer/ptv3ppt/inference/val \
      --scenes scene0011_00 --num_queries 5 --show_all_objects

  # every scene that has predictions, every query (large output!)
  python grounding_visual_results.py \
      --scene_data_dir data/scannetv2/val \
      --lang_json data/scanrefer/ScanRefer_filtered_val.json \
      --preds ptv3ppt=output/ScanRefer/ptv3ppt/inference/val \
      --scenes all --num_queries 0

  # several models overlaid on the same image, specific queries
  python grounding_visual_results.py \
      --scene_data_dir data/scannetv2/val \
      --lang_json data/scanrefer/ScanRefer_filtered_val.json \
      --preds dinov2=out/dinov2/val lseg=out/lseg/val \
      --scenes scene0011_00 --queries 36,0 12,1

Output: {out}/{scene}_obj{object_id}_ann{ann_id}.png
Camera follows the repo's visualize_feature.py recipe (pitch 45, yaw 0,
dist 0.8 x bbox diagonal; auto yaw 180 for scene0031_*). Open3D offscreen if
available, numpy fallback otherwise.
"""

import argparse
import json
import os
import textwrap
from os.path import join, exists

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def torch_load(path):
    """torch.load across PyTorch versions: 2.6+ defaults weights_only=True,
    which breaks these dict-of-numpy-arrays scene .pth files (they fail to
    unpickle under the restricted weights_only loader)."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        return torch.load(path, map_location='cpu')


GT_COLOR = (44, 160, 44)          # green
CONTEXT_COLOR = (185, 185, 185)   # faint gray
CONTEXT_WIDTH = 2.0
MODEL_COLORS = [
    (214, 39, 40),    # red
    (31, 119, 180),   # blue
    (255, 127, 14),   # orange
    (148, 103, 189),  # purple
    (140, 86, 75),    # brown
    (227, 119, 194),  # pink
    (23, 190, 207),   # cyan
]

# AABB corners in canonical bit order (idx = 4*xi + 2*yi + zi); edges differ in 1 bit
BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def corners_from_minmax(mn, mx):
    return np.array([[x, y, z] for x in (mn[0], mx[0])
                     for y in (mn[1], mx[1])
                     for z in (mn[2], mx[2])], dtype=np.float64)


def canonicalize_corners(corners):
    """Accept any 8-corner ordering; rebuild in canonical order from min/max."""
    c = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    return corners_from_minmax(c.min(0), c.max(0))


def inflate_corners(corners, delta):
    """Grow an AABB outward by `delta` on every side."""
    return corners_from_minmax(corners.min(0) - delta, corners.max(0) + delta)


def aabb_iou(corners_a, corners_b):
    a0, a1 = corners_a.min(0), corners_a.max(0)
    b0, b1 = corners_b.min(0), corners_b.max(0)
    inter = np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0)).prod()
    va, vb = (a1 - a0).prod(), (b1 - b0).prod()
    return float(inter / (va + vb - inter + 1e-12))


def is_structural(corners, scene_min, scene_max, frac=0.75):
    """True for wall/floor/ceiling-like instance boxes: their extent nearly
    matches the whole scene along at least two axes (floor/ceiling: x+y,
    walls: one horizontal axis + z). Furniture along one wall only maxes out
    a single axis and is kept."""
    ext = corners.max(0) - corners.min(0)
    scene_ext = np.maximum(scene_max - scene_min, 1e-9)
    return int(np.count_nonzero(ext / scene_ext > frac)) >= 2


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_scene(scene_data_dir, scene):
    """xyz (N,3) float64, rgb (N,3) uint8, gt boxes {object_id: corners(8,3)}."""
    if torch is None:
        raise RuntimeError('PyTorch is required (pip install torch).')
    d = torch_load(join(scene_data_dir, scene + '.pth'))
    xyz = np.asarray(d['xyz'], dtype=np.float64)
    rgb = np.asarray(d['rgb'])
    if rgb.dtype != np.uint8:
        if rgb.min() < -0.01:
            rgb = (rgb + 1.0) * 127.5
        elif rgb.max() <= 1.001:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    gt_boxes = {}
    if d.get('aabb_corner_xyz') is not None:
        for corners, oid in zip(d['aabb_corner_xyz'], d['aabb_obj_ids']):
            gt_boxes[int(oid)] = canonicalize_corners(corners)
    return xyz, rgb, gt_boxes


def load_predictions(pred_dir, scene):
    """{(object_id, ann_id): [corners(8,3), ...]} from a per-scene JSON."""
    fp = join(pred_dir, scene + '.json')
    if not exists(fp):
        raise FileNotFoundError(f'No prediction file {fp}')
    with open(fp) as f:
        items = json.load(f)
    out = {}
    for it in items:
        key = (int(it.get('object_id', -1)), int(it['ann_id']))
        out[key] = [canonicalize_corners(b) for b in np.asarray(it['aabb'],
                                                                dtype=np.float64)]
    return out


def load_language(lang_json):
    """{(scene_id, object_id, ann_id): {'description', 'object_name'}}."""
    if lang_json is None:
        return {}
    with open(lang_json) as f:
        data = json.load(f)
    out = {}
    for it in data:
        key = (it['scene_id'], int(it['object_id']), int(it['ann_id']))
        out[key] = {'description': it.get('description', ''),
                    'object_name': it.get('object_name', ''),
                    'eval_type': it.get('eval_type', '')}
    return out


# --------------------------------------------------------------------------
# Rendering (points + 3D boxes), Open3D or numpy backend
# --------------------------------------------------------------------------
def camera_pose(coords, pitch, yaw):
    center = coords.mean(0)
    diag = np.linalg.norm(coords.max(0) - coords.min(0))
    p, y = np.deg2rad(pitch), np.deg2rad(yaw)
    eye = center + 0.8 * diag * np.array([np.cos(p) * np.cos(y),
                                          np.cos(p) * np.sin(y),
                                          np.sin(p)])
    return center, eye


class O3DRenderer:
    def __init__(self, res=1024, point_size=4.0):
        import open3d as o3d
        from open3d.visualization import rendering
        self.o3d, self.rendering = o3d, rendering
        self.render = rendering.OffscreenRenderer(res, res)
        try:
            self.render.scene.view.set_post_processing(False)
        except Exception:
            pass
        self.render.scene.set_background([1, 1, 1, 1])
        self.mtl = rendering.MaterialRecord()
        self.mtl.shader = 'defaultUnlit'
        self.mtl.point_size = float(point_size)
        self._line_mtls = {}

    def _line_mtl(self, width):
        """One material per line width, so context boxes can be thin while
        GT/model boxes stay thick (a single shared material would force the
        same width on every box)."""
        w = float(width)
        if w not in self._line_mtls:
            m = self.rendering.MaterialRecord()
            m.shader = 'unlitLine'
            m.line_width = w
            self._line_mtls[w] = m
        return self._line_mtls[w]

    def __call__(self, coords, colors_uint8, boxes, pitch=45.0, yaw=0.0):
        o3d = self.o3d
        self.render.scene.clear_geometry()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(colors_uint8.astype(np.float64) / 255.0)
        self.render.scene.add_geometry('pcd', pcd, self.mtl)
        for i, (corners, color, width) in enumerate(boxes):
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(corners)
            ls.lines = o3d.utility.Vector2iVector(np.array(BOX_EDGES))
            col = np.tile(np.asarray(color, dtype=np.float64) / 255.0,
                          (len(BOX_EDGES), 1))
            ls.colors = o3d.utility.Vector3dVector(col)
            self.render.scene.add_geometry(f'box{i}', ls, self._line_mtl(width))
        center, eye = camera_pose(coords, pitch, yaw)
        self.render.setup_camera(60.0, center, eye, [0, 0, 1])
        return np.asarray(self.render.render_to_image())[:, :, :3].astype(np.uint8)


def render_numpy(coords, colors_uint8, boxes, pitch=45.0, yaw=0.0, res=1024,
                 point_px=2, supersample=2, fov_deg=60.0):
    center, cam = camera_pose(coords, pitch, yaw)
    fwd = center - cam
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    R = res * supersample
    half = R / 2.0

    def project(pts):
        rel = pts - cam
        zc = rel @ fwd
        zc = np.maximum(zc, 1e-6)
        xs = (rel @ right) / zc * f
        ys = (rel @ upv) / zc * f
        return xs, ys, zc

    xs, ys, zc = project(coords)
    scale = half * 0.95 / max(np.abs(xs).max(), np.abs(ys).max(), 1e-9)

    def to_px(xs, ys):
        return ((half + xs * scale).astype(np.int64),
                (half - ys * scale).astype(np.int64))

    px, py = to_px(xs, ys)
    order = np.argsort(-zc)  # far -> near
    px, py, cols = px[order], py[order], colors_uint8[order]

    img = np.full((R, R, 3), 255, dtype=np.uint8)
    r = max(1, int(round(point_px * supersample / 2)))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r + 1:
                continue
            img[np.clip(py + dy, 0, R - 1), np.clip(px + dx, 0, R - 1)] = cols

    # boxes on top (not depth-tested; keeps them fully visible)
    for corners, color, width in boxes:
        bx, by, _ = project(np.asarray(corners, dtype=np.float64))
        cx, cy = to_px(bx, by)
        w = max(1, int(round(width * supersample / 2)))
        color = np.asarray(color, dtype=np.uint8)
        for a, b in BOX_EDGES:
            n = int(max(abs(cx[b] - cx[a]), abs(cy[b] - cy[a]))) + 1
            lx = np.linspace(cx[a], cx[b], n).astype(np.int64)
            ly = np.linspace(cy[a], cy[b], n).astype(np.int64)
            for dy in range(-w, w + 1):
                for dx in range(-w, w + 1):
                    img[np.clip(ly + dy, 0, R - 1), np.clip(lx + dx, 0, R - 1)] = color
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
            print(f'[warn] Open3D offscreen unavailable ({e}); using numpy fallback.')
    print('[info] rendering with numpy fallback')
    return lambda c, col, boxes, pitch=45.0, yaw=0.0: render_numpy(
        c, col, boxes, pitch, yaw, res=res,
        point_px=int(round(point_size / 2)) + 1)


def autocrop(img, pad=10, bg=255):
    mask = (img < bg - 2).any(-1)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img[max(ys.min() - pad, 0):ys.max() + pad + 1,
               max(xs.min() - pad, 0):xs.max() + pad + 1]


# --------------------------------------------------------------------------
# Caption strip (query text + colored legend)
# --------------------------------------------------------------------------
def _get_font(size):
    from PIL import ImageFont
    try:
        import matplotlib.font_manager as fm
        return ImageFont.truetype(fm.findfont('DejaVu Sans'), size)
    except Exception:
        return ImageFont.load_default()


def add_caption(img, description, legend):
    """legend: list of (label, (r,g,b))"""
    from PIL import Image, ImageDraw
    W = img.shape[1]
    fsize = max(10, W // 50)
    font = _get_font(fsize)
    line_h = int(fsize * 1.45)
    chars = max(20, int(W / (fsize * 0.52)))
    lines = textwrap.wrap(f'"{description}"', width=chars) if description else []
    strip_h = line_h * (len(lines) + 1) + 16

    canvas = Image.new('RGB', (W, img.shape[0] + strip_h), 'white')
    canvas.paste(Image.fromarray(img), (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = img.shape[0] + 8
    for ln in lines:
        draw.text((10, y), ln, fill=(20, 20, 20), font=font)
        y += line_h
    x = 10
    sq = int(fsize * 0.8)
    for label, color in legend:
        draw.rectangle([x, y + (line_h - sq) // 2, x + sq, y + (line_h - sq) // 2 + sq],
                       fill=tuple(color), outline=(0, 0, 0))
        x += sq + 6
        draw.text((x, y), label, fill=(20, 20, 20), font=font)
        x += int(draw.textlength(label, font=font)) + 18
    return np.asarray(canvas)


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
    ap.add_argument('--scene_data_dir', required=True,
                    help='dir containing M3DRef-CLIP preprocessed {scene}.pth '
                         '(e.g. .../scannetv2/val)')
    ap.add_argument('--scenes', nargs='+', default=None,
                    help="scene ids, or 'all' (or omit) to process every scene "
                         "that has a prediction file in the first --preds dir")
    ap.add_argument('--preds', nargs='+', required=True,
                    help='one or more name=dir; dir holds per-scene {scene}.json '
                         'written by the M3DRef-CLIP test step '
                         '(i.e. {pred_path}/{split})')
    ap.add_argument('--lang_json', default=None,
                    help='ScanRefer_filtered_val.json (for query text)')
    ap.add_argument('--queries', nargs='+', default=None,
                    help='which queries: "object_id,ann_id" pairs (default: '
                         'first --num_queries found in the prediction file)')
    ap.add_argument('--num_queries', type=int, default=5,
                    help='queries per scene when --queries is not given; '
                         '0 or negative = all queries')
    ap.add_argument('--eval_type', choices=['unique', 'multiple', 'all'],
                    default='all',
                    help='only render queries of this ScanRefer difficulty '
                         '(reads the eval_type field of --lang_json, added by '
                         'add_evaluation_labels.py; ignored when --queries '
                         'is given)')
    ap.add_argument('--show_all_objects', action='store_true',
                    help='also draw the other GT object boxes (= the candidate '
                         'pool under use_gt_proposal) as thin gray context')
    ap.add_argument('--keep_structural', action='store_true',
                    help='do not filter wall/floor/ceiling-sized context boxes')
    ap.add_argument('--pitch', type=float, default=45.0)
    ap.add_argument('--yaw', type=float, default=None,
                    help='default 0 (180 for scene0031_*)')
    ap.add_argument('--backend', choices=['auto', 'open3d', 'numpy'], default='auto')
    ap.add_argument('--res', type=int, default=1024)
    ap.add_argument('--point_size', type=float, default=4.0)
    ap.add_argument('--line_width', type=float, default=6.0)
    ap.add_argument('--no_crop', action='store_true')
    ap.add_argument('--out', default='grounding_vis')
    args = ap.parse_args()

    try:
        import PIL  # noqa: F401 -- add_caption() needs this for every image
    except ImportError:
        raise SystemExit('Pillow is required for the caption strip '
                         '(pip install Pillow).')

    model_preds = []
    for it in args.preds:
        if '=' not in it:
            raise SystemExit(f'--preds entries must be name=dir, got: {it}')
        name, d = it.split('=', 1)
        model_preds.append((name.strip(), d.strip()))

    lang = load_language(args.lang_json)
    os.makedirs(args.out, exist_ok=True)
    renderer = get_renderer(args.backend, args.res, args.point_size)

    scenes = args.scenes
    if not scenes or (len(scenes) == 1 and scenes[0].lower() == 'all'):
        pred_dir0 = model_preds[0][1]
        scenes = sorted(os.path.splitext(f)[0] for f in os.listdir(pred_dir0)
                        if f.endswith('.json'))
        if not scenes:
            raise SystemExit(f'No per-scene .json prediction files in {pred_dir0}')
        print(f'[info] processing all {len(scenes)} scenes found in {pred_dir0}')

    for si, scene in enumerate(scenes):
        print(f'=== {scene} ({si + 1}/{len(scenes)}) ===')
        try:
            xyz, rgb, gt_boxes = load_scene(args.scene_data_dir, scene)
        except FileNotFoundError as e:
            print(f'[warn] skipping {scene}: {e}')
            continue
        scene_min, scene_max = xyz.min(0), xyz.max(0)
        preds_per_model = {}
        for name, d in model_preds:
            try:
                preds_per_model[name] = load_predictions(d, scene)
            except FileNotFoundError:
                print(f'[warn] {scene}: no prediction file for {name}')
                preds_per_model[name] = {}
        if not any(preds_per_model.values()):
            print(f'[warn] skipping {scene}: no predictions from any model')
            continue

        # which queries to draw
        if args.queries:
            keys = [tuple(int(x) for x in q.split(',')) for q in args.queries]
        else:
            keys = sorted(set().union(*[set(p.keys())
                                        for p in preds_per_model.values()]))
            if args.eval_type != 'all':
                if not lang:
                    raise SystemExit('--eval_type filtering needs --lang_json')
                typed = [k for k in keys if lang.get((scene,) + k, {})
                         .get('eval_type') == args.eval_type]
                if keys and not typed and not any(
                        lang.get((scene,) + k, {}).get('eval_type')
                        for k in keys):
                    print(f'[warn] {scene}: no "eval_type" field in '
                          f'--lang_json — run dataset/scanrefer/'
                          f'add_evaluation_labels.py first')
                keys = typed
            if args.num_queries > 0:
                keys = keys[:args.num_queries]

        yaw = args.yaw if args.yaw is not None else (
            180.0 if scene.startswith('scene0031_') else 0.0)

        for (oid, aid) in keys:
            boxes, legend = [], []
            if args.show_all_objects:
                n_skipped = 0
                for o, corners in gt_boxes.items():
                    if o == oid:
                        continue
                    if not args.keep_structural and is_structural(
                            corners, scene_min, scene_max):
                        n_skipped += 1
                        continue
                    boxes.append((corners, CONTEXT_COLOR, CONTEXT_WIDTH))
                if n_skipped:
                    print(f'[info] {scene}: hid {n_skipped} wall/floor/ceiling-'
                          f'sized context boxes (--keep_structural to show)')
            gt = gt_boxes.get(oid)
            if gt is not None:
                boxes.append((gt, GT_COLOR, args.line_width))
                legend.append(('GT', GT_COLOR))
            else:
                print(f'[warn] {scene}: no GT box for object_id {oid}')

            drawn_preds = []  # original (uninflated) picks drawn so far
            for m_idx, (name, _) in enumerate(model_preds):
                color = MODEL_COLORS[m_idx % len(MODEL_COLORS)]
                pboxes = preds_per_model[name].get((oid, aid))
                if pboxes is None:
                    print(f'[warn] {scene}: {name} has no prediction for '
                          f'(object_id={oid}, ann_id={aid})')
                    legend.append((f'{name}: n/a', color))
                    continue
                ious = []
                for pb in pboxes:
                    iou = aabb_iou(pb, gt) if gt is not None else None
                    coincides = (iou is not None and iou > 0.99) or any(
                        aabb_iou(pb, prev) > 0.99 for prev in drawn_preds)
                    draw_box = pb
                    if coincides:
                        # this pick sits exactly on the GT box and/or on a
                        # previous model's pick (use_gt_proposal: all models
                        # choose from the same candidate set) — inflate a
                        # touch so it shows as a concentric outline instead
                        # of hiding the box underneath
                        delta = 0.02 * np.linalg.norm(pb.max(0) - pb.min(0)) \
                            * (1 + m_idx)
                        draw_box = inflate_corners(pb, delta)
                    drawn_preds.append(pb)
                    boxes.append((draw_box, color, args.line_width))
                    if iou is not None:
                        ious.append(iou)
                if ious:
                    best = max(ious)
                    tag = (f'{name} 100%' if best > 0.99
                           else f'{name} 0%')
                else:
                    tag = name
                legend.append((tag, color))

            img = renderer(xyz, rgb, boxes, pitch=args.pitch, yaw=yaw)
            if not args.no_crop:
                img = autocrop(img)

            info = lang.get((scene, oid, aid), {})
            desc = info.get('description', '')
            etype = info.get('eval_type', '')
            tag_bits = [b for b in (info.get('object_name', ''), etype) if b]
            name_str = f" [{' | '.join(tag_bits)}]" if tag_bits else ''
            img = add_caption(img, desc + name_str if desc else '', legend)
            suffix = f'_{etype}' if etype else ''
            save_png(img, join(args.out,
                               f'{scene}_obj{oid}_ann{aid}{suffix}.png'))


if __name__ == '__main__':
    main()