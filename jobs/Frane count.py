import os

# Your specific paths
base_2d_path = "/home/fferrabe/Thesis26/data/openscene_format/scannet_2d"
train_split_file = "/home/fferrabe/Thesis26/data/openscene_format/scannet_3d/meta_data/scannetv2_train.txt"
val_split_file = "/home/fferrabe/Thesis26/data/openscene_format/scannet_3d/meta_data/scannetv2_val.txt"

def count_frames_for_split(split_file, base_path):
    total_frames = 0
    missing_scenes = 0
    
    if not os.path.exists(split_file):
        print(f"Error: Split file not found at {split_file}")
        return 0, 0

    with open(split_file, 'r') as f:
        # Strip whitespace and ignore empty lines
        scenes = [line.strip() for line in f.readlines() if line.strip()]
    
    for scene in scenes:
        # Path to the color folder for this specific scene
        color_path = os.path.join(base_path, scene, "color")
        
        if os.path.exists(color_path):
            # Count files in the color directory
            count = len(os.listdir(color_path))
            total_frames += count
        else:
            missing_scenes += 1
            
    return total_frames, len(scenes), missing_scenes

# Run counts
train_total, train_expected, train_missing = count_frames_for_split(train_split_file, base_2d_path)
val_total, val_expected, val_missing = count_frames_for_split(val_split_file, base_2d_path)

# Output Results
print("-" * 30)
print(f"TRAIN SPLIT RESULTS:")
print(f"  Total Frames:    {train_total}")
print(f"  Scenes Found:    {train_expected - train_missing}/{train_expected}")
if train_missing > 0:
    print(f"  Warning: {train_missing} scenes were not found in the 2d folder.")

print("-" * 30)
print(f"VAL SPLIT RESULTS:")
print(f"  Total Frames:    {val_total}")
print(f"  Scenes Found:    {val_expected - val_missing}/{val_expected}")
if val_missing > 0:
    print(f"  Warning: {val_missing} scenes were not found in the 2d folder.")
print("-" * 30)
