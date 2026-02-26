import os
import glob

# Define directories
base_path = '/home/pinaka/GANav-offroad/data/avmi_ugv/annotations'
dirs = ['train', 'val', 'test']

for dir_name in dirs:
    dir_path = os.path.join(base_path, dir_name)
    
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        continue
    
    # Find all *_mask.png files
    mask_files = glob.glob(os.path.join(dir_path, '*_mask.png'))
    
    print(f"\nProcessing {dir_name}/ directory...")
    print(f"Found {len(mask_files)} mask files")
    
    for old_path in mask_files:
        # Get new filename without '_mask'
        new_path = old_path.replace('_mask.png', '.png')
        
        # Rename file
        os.rename(old_path, new_path)
        print(f"  Renamed: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
    
    print(f"✓ Completed {dir_name}/")

print("\n✓ All masks renamed successfully!")
