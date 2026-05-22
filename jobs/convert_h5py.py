import os
import torch
import h5py
import numpy as np
from tqdm import tqdm

# Paths
input_dir = "/home/fferrabe/Thesis26/data/openscene_format/scannet_3d/lexicon3d/dinov2/dinov2_features"
output_file = "/home/fferrabe/Thesis26/data/openscene_format/scannet_3d/lexicon3d/dinov2/dinov2_feats.hdf5"

def main():
    print(f"Scanning directory: {input_dir}")
    files = [f for f in os.listdir(input_dir) if f.endswith('.pth') or f.endswith('.pt')]
    print(f"Found {len(files)} feature files. Packing into HDF5...")
    
    # CHANGED: 'a' mode opens the file for appending (creates it if it doesn't exist)
    with h5py.File(output_file, 'a') as h5f:
        for filename in tqdm(files):
            scene_id = filename.split('.')[0]
            
            # CHANGED: If the scene is already in the HDF5 file, skip it
            if scene_id in h5f:
                continue
                
            file_path = os.path.join(input_dir, filename)
            feat_data = torch.load(file_path, map_location='cpu')
            
            if isinstance(feat_data, dict):
                key = 'feat' if 'feat' in feat_data else list(feat_data.keys())[0]
                feat_array = feat_data[key].numpy()
            else:
                feat_array = feat_data.numpy()
            
            h5f.create_dataset(scene_id, data=feat_array, dtype=np.float32, compression="gzip")
            
    print(f"Success! HDF5 file saved to: {output_file}")

if __name__ == "__main__":
    main()