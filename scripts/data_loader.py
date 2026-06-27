import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset

def transform_to_agent_centric(past,future):

    # 1. Estimation of origin
    origin = past[-1]

    # 2. Translate the positions
    past_trans = past - origin
    future_trans = future - origin

    # 3. calculate the angle theta at origin frame
        # taking vector between the second-to-last and last point
    delta_x = past[-1,0] - past[-2,0]
    delta_y = past[-1,1] - past[-2,1]
    theta = np.arctan2(delta_y,delta_x)

    # 4. Create Rotation Matrix (Standard 2D rotation)
    cos_t, sin_t = np.cos(-theta), np.sin(-theta) # we are doing a inverse rotation
    # ex. car heading is 45 degree counter-clockwise,
    # our goal is to force the car to face exactly $0^\circ$ (flat along the X-axis).
    # To get the map from $+45^\circ$ down to $0^\circ$, we have to spin the entire world in the opposite direction.
    R = np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])

    # 5. Rotate both past and future trajectories
    # Matrix multiplication dot product applies rotation to all rows
    past_normalized = np.dot(past_trans,R.T)
    future_normalized = np.dot(future_trans,R.T)
    return past_normalized,future_normalized, origin, theta

def load_sequence(file_path):
    df = pd.read_csv(file_path)
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].sort_values('TIMESTAMP')

    coords = agent_df[["X","Y"]].values
    if len(coords) < 50:
        return None
    past = coords[:20]
    future = coords[20:50]
    # normalization
    past, future, origin, theta = transform_to_agent_centric(past,future)
    v = np.gradient(past, 0.1, axis=0)
    # adding velocity feature
    past = np.hstack((past,v))
    return past,future, origin, theta


class ArgoDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # High-speed binary loading directly into memory space
        data = torch.load(self.files[idx])
        return data['past'], data['future'], data['origin'], data['theta']