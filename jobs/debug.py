import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from safetensors.torch import load_file
import os
import torch
import imageio
import argparse
import transformers
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import sys
sys.path.append('..')

import torchvision.transforms as transforms
from PIL import Image

# 1. Use the README's suggested path
import torch
from transformers import AutoModel, AutoImageProcessor

# 1. Load the model using the Hugging Face logic
# This will automatically map your 'embeddings.' and 'layer.' keys correctly
model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
model.to("cuda").eval()

def debug_step(image_path, model, transform):
    # 1. Load and Transform Image
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).cuda()
    
    # 2. Get Features (Hugging Face style)
    with torch.no_grad():
        # output_hidden_states=True gives us access to all layers
        outputs = model(input_tensor, output_hidden_states=True)
        
        # outputs.hidden_states is a list of all layers
        # index -1 is the last layer, -2 is the penultimate
        feat = outputs.hidden_states[-2] # [1, 305, 1024]
    
    total_tokens = feat.shape[1]
    print(f"--- DINOv3 (HF) Token Audit ---")
    print(f"Input Tensor Shape: {input_tensor.shape}")
    print(f"Total Tokens Detected: {total_tokens}")
    
    # 305 tokens = 1 CLS + 4 Registers + 300 Patches
    expected_spatial = 300 
    extra_tokens = total_tokens - expected_spatial
    
    print(f"Expected Spatial Tokens (15x20): {expected_spatial}")
    print(f"Extra Tokens (CLS/Registers): {extra_tokens}")
    
    # 3. Extract and Reshape
    # Slicing from the end ensures we get the 300 spatial patches
    spatial_feat = feat[:, -300:, :] 
    grid_feat = spatial_feat.reshape(15, 20, 1024).cpu().numpy()
    
    # 4. Visualize with PCA
    pca = PCA(n_components=3)
    flat_feat = grid_feat.reshape(-1, 1024)
    pca_feat = pca.fit_transform(flat_feat)
    
    # Normalize to 0-1 for plotting
    pca_feat = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min())
    pca_img = pca_feat.reshape(15, 20, 3)
    
    # 5. Plot Comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image (Resized)")
    plt.imshow(img.resize((320, 240)))
    
    plt.subplot(1, 2, 2)
    plt.title("DINOv3 PCA ( Penultimate Layer)")
    plt.imshow(pca_img, interpolation='nearest')
    plt.show()

    return extra_tokens

transform = transforms.Compose([
    # Resize to match your 15x20 patch grid (240/16=15, 320/16=20)
    transforms.Resize((240, 320)), 
    transforms.ToTensor(),
    # DINOv3 official ImageNet normalization
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- Run Debug ---
# Replace with a valid path from your ScanNet 2D folder
debug_step('/home/fferrabe/Thesis26/data/openscene_format/scannet_2d/scene0011_00/color/0.jpg', model, transform)