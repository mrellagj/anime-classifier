import os
import shutil
import random
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
    
def process_directories(start_path:Path, target_path:Path):
    
    target_path.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(start_path):
        root_path = Path(root)
        if 'art' in dirs:
            arts_path = root_path / 'art'
            try:
                art_files = list(arts_path.iterdir())
                if len(art_files) < 10:
                    print(f"Skipping {arts_path} due to insufficient files.")
                    continue                    
                    
                for item in art_files:
                    dest_path = target_path / root_path.name / item.name
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if dest_path.exists():
                        print(f"Destination already exists: {dest_path}, skipping.")
                        continue
                    
                    os.link(str(item), str(dest_path))

                print(f"Extracted contents from: {arts_path}")
            except Exception as e:
                print(f"Error processing {arts_path}: {e}")

def process_directories_full(start_path:Path, target_path:Path):
    
    target_path.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(start_path):
        root_path = Path(root)  
        
        total_files = 0
        for check_subdir in ['art', 'picture']:
            if check_subdir in dirs:
                check_path = root_path / check_subdir
                try:
                    total_files += len(list(check_path.iterdir()))
                except Exception:
                    continue
        
        if total_files < 10:
            print(f"Skipping {root_path} due to insufficient files ({total_files} found).")
            continue
                
        for subdir in ['art', 'picture']:            
            if subdir in dirs:
                src_path = root_path / subdir
                try:
                    src_files = list(src_path.iterdir())          
                        
                    for item in src_files:
                        dest_path = target_path / root_path.name / item.name
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        if dest_path.exists():
                            print(f"Destination already exists: {dest_path}, skipping.")
                            continue
                        
                        os.link(str(item), str(dest_path))

                    print(f"Extracted contents from: {src_path}")
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")
                    
def split_and_move_files(source_dir:Path, target_dir:Path, split_ratio=0.7): 
    
    target_dir.mkdir(parents=True, exist_ok=True)
    # Create train, val, and test directories
    for split in ['train', 'val', 'test']:
        (target_dir / split).mkdir(exist_ok=True)

    # Get all subdirectories (classes)
    classes = [d for d in source_dir.iterdir() if d.is_dir()]
    
    for class_path in classes:
        # Ignore mp4 and gif files
        files = [f for f in class_path.iterdir() if f.is_file() and not f.suffix.lower() in ['.mp4', '.gif']]
        random.shuffle(files)
        
        # Calculate split sizes
        n_files = len(files)
        n_train = int(n_files * split_ratio)
        n_val = int(n_files * (1 - split_ratio) / 2)
        # Split files
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Create class directories in each split
        for split in ['train', 'val', 'test']:
            (target_dir / split / class_path.name).mkdir(exist_ok=True)
        
        # Move files to respective directories
        for f in train_files:
            os.link(str(f), str(target_dir / 'train' / class_path.name / f.name))
        for f in val_files:
            os.link(str(f), str(target_dir / 'val' / class_path.name / f.name))
        for f in test_files:
            os.link(str(f), str(target_dir / 'test' / class_path.name / f.name))

def compute_mean_and_std(target_dir:Path):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch
    print("Computing mean and standard deviation for images...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor()
    ])
    
    means = []
    stds = []
    
    dataset = datasets.ImageFolder(root=target_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=256, num_workers=4, pin_memory=True, shuffle=False)
    
    print("Calculating mean and std for the dataset...")
    print(f"Dataset size: {len(dataset)}")
    for batch in tqdm(dataloader):
        images = batch[0]  # Get the image batch
        batch_mean = torch.mean(images, dim=[0, 2, 3])  # Mean across batch and spatial dims
        batch_std = torch.std(images, dim=[0, 2, 3])    # Std across batch and spatial dims
        
        means.append(batch_mean.numpy())
        stds.append(batch_std.numpy())

    overall_mean = np.mean(means, axis=0)
    overall_std = np.mean(stds, axis=0)
    
    print(f"Mean: {overall_mean}, Std: {overall_std}")

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Process anime directories and files.")
    parser.add_argument('--source', type=str, default='/home/gjmrem/Yandex.Disk/ /Anime', help='Source directory containing anime data')
    parser.add_argument('--target', type=str, default='/home/gjmrem/datasets/.anime', help='Target directory for processed data')
    return parser.parse_args()

def main():
    args = parse_arguments()
    source_dir = Path(args.source)
    target_dir = Path(args.target)
    
    # Uncomment the desired processing function
    # process_directories(source_dir, target_dir/'raw')
    # process_directories_full(source_dir, target_dir/'raw')
    split_and_move_files(target_dir/'raw', target_dir/'processed')
    # compute_mean_and_std(target_dir/'raw')
    
if __name__ == "__main__":
    main()