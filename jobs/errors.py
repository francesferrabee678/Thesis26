import torch

path = "/home/fferrabe/Thesis26/data/openscene_format/scannet_3d/lexicon3d/dinov2/dinov2_features/scene0000_00.pt"

x = torch.load(path)

print(type(x))

if isinstance(x, dict):
    print(x.keys())

    for k, v in x.items():
        if torch.is_tensor(v):
            print(f"\n{k}:")
            print("  shape:", v.shape)
            print("  dtype:", v.dtype)
            print("  mean :", v.float().mean().item())
            print("  std  :", v.float().std().item())
            print("  min  :", v.float().min().item())
            print("  max  :", v.float().max().item())

            if v.ndim >= 2:
                norms = v.float().norm(dim=-1)
                print("  norm mean:", norms.mean().item())
                print("  norm std :", norms.std().item())

            print("  has nan:", torch.isnan(v).any().item())
            print("  has inf:", torch.isinf(v).any().item())