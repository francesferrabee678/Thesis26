import torch
import numpy as np

def debug_alignment(v2_path, v3_path):
    # Load data
    d2 = torch.load(v2_path)
    d3 = torch.load(v3_path)
    
    f2 = torch.nn.functional.normalize(d2['feat'].float(), dim=-1)
    f3 = torch.nn.functional.normalize(d3['feat'].float(), dim=-1)
    
    print(f"V2 Points: {f2.shape[0]} | V3 Points: {f3.shape[0]}")
    
    # If sizes don't match, we take the intersection
    min_pts = min(f2.shape[0], f3.shape[0])
    f2, f3 = f2[:min_pts], f3[:min_pts]

    # Test 1: Direct Cosine Similarity
    score = (f2 * f3).sum(dim=-1).mean().item()
    print(f"Direct Alignment Score: {score:.4f}")

    # Test 2: Channel-wise check (Are the features just scaled?)
    std3 = f3.std().item()
    std2 = f2.std().item()
    print(f"V3 Std: {std3:.4f} | V2 Std: {std2:.4f}")

if __name__ == "__main__":
    V2 = '/home/fferrabe/Thesis26/data/openscene_format/scannet_3d/lexicon3d/dinov2/dinov2_features/scene0000_00.pt'
    V3 = '/home/fferrabe/Thesis26/data/openscene_format/scannet_3d/lexicon3d/dinov3/dinov2_features/scene0000_00.pt'
    debug_alignment(V2, V3)