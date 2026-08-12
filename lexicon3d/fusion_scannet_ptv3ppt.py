import os
import torch
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import trimesh

import sys
sys.path.append('..')
# Add Pointcept to path if it's not installed as a package
sys.path.append('/home/fferrabe/Thesis26/lexicon3d/Pointcept') 

from fusion_util import save_fused_feature_with_locs, adjust_intrinsic, make_intrinsic

# Pointcept imports for PTv3
from pointcept.models import build_model
from pointcept.utils.config import DictAction, Config
import pointcept.datasets.transform as T


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of PointTransformerV3 on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='val', help='split: "train"| "val"')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--prefix', type=str, default='ptv3ppt', help='prefix for the output file')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args



def main(args):   
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    scan_dir = '/home/fferrabe/Thesis26/data/scannet_raw/scans/'

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range

    args.n_split_points = 2000000

    ########################################
    ##### Load the PTv3 (Pointcept) Model ##
    ########################################

    cfg = Config.fromfile('/home/fferrabe/Thesis26/lexicon3d/Pointcept/weights/scannet-semseg-pt-v3m1-1-ppt-extreme/config.py')
    model = build_model(cfg.model).cuda()
    checkpoint = torch.load("/home/fferrabe/Thesis26/lexicon3d/Pointcept/weights/scannet-semseg-pt-v3m1-1-ppt-extreme/model/model_best.pth", map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
            
        if not k.startswith('backbone.') and k not in ['logit_scale', 'class_embedding', 'proj_head']:
            k = f"backbone.{k}"
            
        new_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    args.evaluator = model
    device = torch.device('cpu')


    voxelize_transform = T.Compose([
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=0.02, 
            hash_type="fnv", 
            mode="test", 
            keys=("coord", "color", "normal"), 
            return_grid_coord=True,
            return_inverse=True  
        )
    ])

    format_transform = T.Compose([
        dict(type="CenterShift", apply_z=False),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "color", "normal", "inverse"),
            offset_keys_dict=dict(offset="coord"),
            feat_keys=("color", "normal")
        )
    ])

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]


    # obtain scene names from scan_dir
    data_paths = sorted(glob(join(scan_dir, "*/*_vh_clean_2.ply")))
    scene_names = [data_path.split('/')[-2] for data_path in data_paths]

    data_3d_paths = sorted(glob('/home/fferrabe/Thesis26/data/openscene_format/scannet_3d/*/*.pth'))
    
    # For val_loader, only iterate id_range[0] to id_range[1]    
    for i, data_3d_path in enumerate(tqdm(data_3d_paths)):        
        scene_id = os.path.basename(data_3d_path).split('.')[0]
        if id_range is not None and \
           (i<id_range[0] or i>id_range[1]):
            print('skip ', i, scene_id)
            continue


        if exists(join(out_dir, args.prefix+'_points', scene_id + '.npy')):
            print(scene_id + '.npy already exists, skip!')
            continue
            
        with torch.no_grad():
            data = torch.load(data_3d_path)
            locs_in = data[0]
            colors_in = data[1] if len(data) > 1 else np.zeros_like(locs_in)
            coord_np=(locs_in.cpu().numpy() if torch.is_tensor(locs_in) else locs_in).copy()
            coord_np_original=coord_np.copy()

            base_scene_name = scene_id.replace('_vh_clean_2', '')
            ply_path = join(scan_dir, base_scene_name, f"{base_scene_name}_vh_clean_2.ply")
            mesh = trimesh.load(ply_path, process=False)

            point_dict = {
                "coord": coord_np,
                "color": ((colors_in.cpu().numpy() if torch.is_tensor(colors_in) else colors_in) + 1.0) / 2.0,
                "normal" : mesh.vertex_normals
            }

            point_list = voxelize_transform(point_dict)
            point_dict_unpacked = point_list[0]
            point = format_transform(point_dict_unpacked)

            for key in point.keys():
                if isinstance(point[key], torch.Tensor):
                    point[key] = point[key].cuda(non_blocking=True)

            inverse_map = point["inverse"]
            point["condition"] = ["ScanNet"]

            point = model(point)

            feats = point.feat[inverse_map].cpu()

            n_points = locs_in.shape[0]

            assert locs_in.shape[0] == feats.shape[0], (
                f"Mismatch! coords={locs_in.shape[0]} features={feats.shape[0]}"
            )

            point_ids_all = torch.arange(n_points, device=device)
            
            assert np.allclose(
                locs_in.cpu().numpy() if torch.is_tensor(locs_in) else locs_in,
                coord_np_original  # the .copy() you made before any transform touched it
            ), f"locs_in was mutated during transform pipeline for {scene_id}!"

            folder_scene_id = scene_id.replace('_vh_clean_2', '')
            save_fused_feature_with_locs(feats, point_ids_all, locs_in, n_points, out_dir, folder_scene_id, args)
            
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = get_args()
    for h in args.hparams:
        if '=' in h:
            k, v = h.split('=')
            setattr(args, k, v)
    print("Arguments:")
    print(args)

    main(args)