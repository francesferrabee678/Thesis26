import os
import torch
from torch import nn
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import sys
sys.path.append('..')

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from fusion_util import (
    PointCloudToImageMapper,
    save_fused_feature_with_locs,
    adjust_intrinsic,
    make_intrinsic,
)



class UniScene3D(nn.Module):

    def __init__(self, uniscene3d_root, uniscene3d_ckpt=None,
                 image_size=224, patch_size=16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size    

        src_dir = join(uniscene3d_root, "src")
        if src_dir not in sys.path:
            sys.path.append(src_dir)
        from common.misc import build_fgclip_model_from_local_code_with_hf_weights

        model_root = join(src_dir, "fg-clip")
        model = build_fgclip_model_from_local_code_with_hf_weights(model_root)

        if uniscene3d_ckpt is not None:
            self._load_uniscene3d_ckpt(model, uniscene3d_ckpt)

        self.model = model.eval().to(torch.float32)
        self.feat_dim = self.model.visual_projection.out_features

    @staticmethod
    def _load_uniscene3d_ckpt(model, ckpt_path):
        if os.path.isdir(ckpt_path):
            from safetensors.torch import load_file
            weight_files = sorted(glob(join(ckpt_path, "model*.safetensors")))
            if not weight_files:
                raise FileNotFoundError(
                    f"No model*.safetensors in {ckpt_path}")
            state = {}
            for wf in weight_files:
                state.update(load_file(wf, device="cpu"))
        elif ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(ckpt_path, device="cpu")
        elif ckpt_path.endswith((".pth", ".pt")):
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
        else:
            raise FileNotFoundError(f"Unsupported checkpoint path: {ckpt_path}")

        def strip_wrappers(key):
            for pfx in ("module.", "model.", "target_model."):
                if key.startswith(pfx):
                    key = key[len(pfx):]
            return key
        state = {strip_wrappers(k): v for k, v in state.items()}
        if any("pm_encoder." in k for k in state):
            cleaned = {k.split("pm_encoder.", 1)[1]: v
                       for k, v in state.items() if "pm_encoder." in k}
        else:
            cleaned = state


    def forward(self, color_pm):
        feat = self.model.get_image_dense_features(color_pm)     
        B = feat.shape[0]
        feat = feat.float().reshape(B, self.grid, self.grid, self.feat_dim)
        feat = feat.permute(0, 3, 1, 2).contiguous()             
        return feat


#####
#Point-map construction
######
def build_pointmap(depth, intrinsic, pose, xy_center=None):
    # Back-project a depth image into a world-frame XYZ
    H, W = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    us = np.arange(W)
    vs = np.arange(H)
    uu, vv = np.meshgrid(us, vs)

    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    cam = np.stack([x, y, z, np.ones_like(z)], axis=-1) 

    world = cam @ pose.T                                
    world = world[..., :3]

    if xy_center is not None:
        world[..., 0] -= xy_center[0]
        world[..., 1] -= xy_center[1]

    invalid = (depth <= 0)
    world[invalid] = 0.0
    return world.astype(np.float32)


def get_args():
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of UniScene3D on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='val', help='split: "train"| "val"')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--prefix', type=str, default='uniscene3d', help='prefix for the output file')
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


###############
#Per-scene processing
###############
def process_one_scene(data_path, out_dir, args):
    scene_id = data_path.split('/')[-1].split('_vh')[0]

    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale
    evaluator = args.evaluator
    transform = args.transform

    # load 3D data (point cloud)
    locs_in = torch.load(data_path)[0]
    n_points = locs_in.shape[0]

    if exists(join(out_dir, args.prefix + '_points', scene_id + '.npy')):
        print(scene_id + '.pt already exists, skip!')
        return 1

    scene = join(args.data_root_2d, scene_id)
    img_dirs = sorted(glob(join(scene, 'color/*')),
                      key=lambda x: int(os.path.basename(x)[:-4]))
    num_img = len(img_dirs)
    device = torch.device('cpu')

    first_depth_path = img_dirs[0].replace('color', 'depth').replace('jpg', 'png')
    first_depth_arr = imageio.v2.imread(first_depth_path)
    depth_h, depth_w = first_depth_arr.shape
    pm_intr = adjust_intrinsic(
        args.pm_intrinsic.copy(),  
        intrinsic_image_dim=args.pm_base_dim,
        image_dim=(depth_w, depth_h)
    )
    locs_np = locs_in.numpy() if torch.is_tensor(locs_in) else np.asarray(locs_in)
    xy_center = locs_np[:, :2].mean(0).astype(np.float32)



    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, feat_dim), device=device)

    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    for img_id, img_dir in enumerate(tqdm(img_dirs)):

        # load pose
        posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
        pose = np.loadtxt(posepath)

        # skip corrupt / degenerate poses
        if not np.isfinite(pose).all():
            continue

        # load depth and convert to meters
        depth = imageio.v2.imread(
            img_dir.replace('color', 'depth').replace('jpg', 'png')) / depth_scale

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
        if mapping[:, 3].sum() == 0:
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask


        image = Image.open(img_dir).convert('RGB')
        image = transform(image)   


        pm = build_pointmap(depth, pm_intr, pose, xy_center=xy_center)
        pm = torch.from_numpy(pm).permute(2, 0, 1).unsqueeze(0)
        pm = torch.nn.functional.interpolate(
            pm, size=(args.image_size, args.image_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        color_pm = torch.cat([image, pm], dim=0).unsqueeze(0).to('cuda')  
        with torch.no_grad():
            feat_2d = evaluator(color_pm)                        
        feat_2d = torch.nn.functional.interpolate(
            feat_2d, size=(240, 320), mode='bicubic', align_corners=False
        ).squeeze(0) 

        feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0).to(device)

        counter[mask != 0] += 1
        sum_features[mask != 0] += feat_2d_3d[mask != 0]

    counter[counter == 0] = 1e-5
    feat_bank = sum_features / counter
    point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

    save_fused_feature_with_locs(feat_bank, point_ids, locs_in, n_points,
                                 out_dir, scene_id, args)


#############
#  Main
#############
def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    img_dim = (320, 240)
    depth_scale = 1000.0
    fx = 577.870605
    fy = 577.870605
    mx = 319.5
    my = 239.5
    transforms_mean = [0.48145466, 0.4578275, 0.40821073]
    transforms_std = [0.26862954, 0.26130258, 0.27577711]
    visibility_threshold = 0.25

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10
    args.image_size = 224

    split = args.split
    data_dir = args.data_dir

    data_root = join(data_dir, 'scannet_3d')
    data_root_2d = join(data_dir, 'scannet_2d')
    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range

    if split == 'train':
        args.n_split_points = 300000
    else:
        args.n_split_points = 2000000

    ##############################
    ##### load UniScene3D model ##
    uniscene3d_root = '/home/fferrabe/Thesis26/lexicon3d/UniScene3D'
    uniscene3d_ckpt = '/home/fferrabe/Thesis26/lexicon3d/UniScene3D/checkpoints/uniscene3d'

    model = UniScene3D(uniscene3d_root=uniscene3d_root,
                       uniscene3d_ckpt=uniscene3d_ckpt).cuda()
    args.evaluator = model
    args.feat_dim = model.feat_dim

    args.transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size),
                           interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=transforms_mean, std=transforms_std),
    ])


    args.pm_base_dim = [640, 480]
    args.pm_intrinsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)

    intrinsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
    intrinsic = adjust_intrinsic(intrinsic, intrinsic_image_dim=[640, 480],
                                 image_dim=img_dim)
    args.point2img_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=intrinsic,
        visibility_threshold=visibility_threshold,
        cut_bound=args.cut_num_pixel_boundary)

    data_paths = sorted(glob(join(data_root, split, '*.pth')))
    total_num = len(data_paths)

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]),
                    int(process_id_range[0].split(',')[1])]

    for i in trange(total_num):
        if id_range is not None and (i < id_range[0] or i > id_range[1]):
            print('skip ', i, data_paths[i])
            continue
        process_one_scene(data_paths[i], out_dir, args)


if __name__ == "__main__":
    args = get_args()
    for h in args.hparams:
        if '=' in h:
            k, v = h.split('=')
            setattr(args, k, v)
    print("Arguments:")
    print(args)
    main(args)