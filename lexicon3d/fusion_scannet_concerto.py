import os
import torch
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
# add the parent directory to the path
import sys
sys.path.append('..')
from PIL import Image
import trimesh
from fusion_util import save_fused_feature_with_locs, adjust_intrinsic, make_intrinsic
import concerto

def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of Concerto on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='val', help='split: "train"| "val"')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--prefix', type=str, default='concerto', help='prefix for the output file')


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


    ##############################
    ##### load the Concerto model ####

    model = concerto.model.load("concerto_small", repo_id="Pointcept/Concerto").cuda()
    model.eval()
    args.evaluator = model
    device = torch.device('cpu')

    transform = concerto.transform.default()

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
            print(scene_id +'.pt' + ' already exists, skip!')
            continue

        with torch.no_grad():
            data = torch.load(data_3d_path)
            locs_in = data[0]
            colors_in = data[1] if len(data) > 1 else np.zeros_like(locs_in)
            normals_in = np.zeros_like(locs_in)

            base_scene_name = scene_id.replace('_vh_clean_2', '')
            ply_path = join(scan_dir, base_scene_name, f"{base_scene_name}_vh_clean_2.ply")
            mesh = trimesh.load(ply_path, process=False)
            inverse_map = torch.from_numpy(locs_in) if isinstance(locs_in, np.ndarray) else locs_in
            
            point_dict = {
                "coord": (locs_in.cpu().numpy() if torch.is_tensor(locs_in) else locs_in).copy(),
                "color": ((colors_in.cpu().numpy() if torch.is_tensor(colors_in) else colors_in) + 1.0) / 2.0,
                "normal": mesh.vertex_normals
            }

            point = transform(point_dict)

            for key in point.keys():
                if isinstance(point[key], torch.Tensor):
                    point[key] = point[key].cuda(non_blocking=True)
                    
            point = model(point)

            for _ in range(2):
                assert "pooling_parent" in point.keys()
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat= torch.cat([parent.feat , point.feat[inverse]], dim=-1)
                point = parent

            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = point.feat[inverse]
                point = parent

            # use inverse_map to get the original point cloud
            n_points = inverse_map.shape[0]
            feats = point.feat[point.inverse].cpu()

            print(f"\nScene: {scene_id}")
            print(f"Coords shape: {locs_in.shape}")
            print(f"Features shape: {feats.shape}")
            print(f"Num coords: {locs_in.shape[0]}")
            print(f"Num features: {feats.shape[0]}")
            print(f"Feature dim: {feats.shape[1]}")

            assert locs_in.shape[0] == feats.shape[0], (
                f"Mismatch! coords={locs_in.shape[0]} "
                f"features={feats.shape[0]}"
    )

            point_ids_all = torch.arange(n_points, device=device)
            
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
