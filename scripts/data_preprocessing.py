
import os
import pandas as pd
import torch
from tqdm import tqdm
from data_loader import load_sequence  # Assumes load_sequence is in your data_loader.py

def pre_process_dataset(source_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]
    print(f"Converting {len(files)} files from {source_folder} to binary tensors...")
    
    for f in tqdm(files):
        file_path = os.path.join(source_folder, f)
        data = load_sequence(file_path)
        if data is not None:
            past, future, origin, theta = data
            payload = {
                'past': torch.tensor(past, dtype=torch.float32),
                'future': torch.tensor(future, dtype=torch.float32),
                'origin': torch.tensor(origin, dtype=torch.float32),
                'theta': torch.tensor(theta, dtype=torch.float32)
            }
            output_path = os.path.join(output_folder, f.replace('.csv', '.pt'))
            torch.save(payload, output_path)


train_src = "/data/train/data"
train_dest = "/data/tensor_data/train"
pre_process_dataset(train_src, train_dest)