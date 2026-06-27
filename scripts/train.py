import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_loader import ArgoDataset
from model import LSTMPredictor
import torch.nn as nn
import torch.optim as optim
import time
start_time = time.time()
model_path = "../models/model.pth"
# training dataset
train_folder = "/content/tensor_data/train"
val_folder = "/content/tensor_data/val"
batch_size = 1024
resume_training = False  # Set to False if you ever want to start a brand-new training run


dataset = ArgoDataset(train_folder)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

train_data_loading_time = time.time() - start_time
print(f" Training data loaded in {train_data_loading_time:.2f} [s]")

# validation dataset
val_dataset = ArgoDataset(val_folder)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
val_data_loading_time = time.time() - start_time
print(f" Validation data loaded in {val_data_loading_time:.2f} [s]")

model = LSTMPredictor()
#Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
# mode='min': We want the validation loss to go down.
# factor=0.5: Cut the learning rate in half when triggered.
# patience=3: Wait for 3 epochs of NO improvement before cutting the LR.
loss_func = nn.SmoothL1Loss()

def de_normalize(pred_normalized,origin, theta):
    # pushing everything to same device as input data, this is written after pytorch lost the track of devices
    target_device = pred_normalized.device
    # 2. Explicitly force theta, cos, and sin onto that exact same device
    theta = theta.to(target_device)
    cos_t = torch.cos(theta).to(target_device)
    sin_t = torch.sin(theta).to(target_device)
    # Also ensure origin matches the target device
    origin = origin.to(target_device)
    # Reconstruct the forward rotation matrix batch-wise
    # pred_normalized: (B, 30, 2)
    x_rotated = pred_normalized[..., 0] * cos_t.unsqueeze(1) - pred_normalized[..., 1] * sin_t.unsqueeze(1)
    y_rotated = pred_normalized[..., 0] * sin_t.unsqueeze(1) + pred_normalized[..., 1] * cos_t.unsqueeze(1)

    x_global = x_rotated + origin[..., 0].unsqueeze(1)
    y_global = y_rotated + origin[..., 1].unsqueeze(1)

    return torch.stack([x_global, y_global], dim=-1)

def ADE(pred, true):
    return ((pred - true) ** 2).sum(dim=2).sqrt().mean()

def FDE(pred, true):
    return ((pred[:, -1] - true[:, -1]) ** 2).sum(dim=1).sqrt().mean()

def validation_loss(model, val_dataloader, loss_func):
    model.eval()
    total_loss, total_ade, total_fde = 0, 0, 0
    num_batches = len(val_dataloader)
    with torch.no_grad():
        for batch_features, batch_targets, origin_val, theta_val in val_dataloader:
            batch_features, batch_targets, origin_val, theta_val = batch_features.to(device), batch_targets.to(device), origin_val.to(device), theta_val.to(device)
            predictions = model(batch_features)
            loss_val = loss_func(predictions, batch_targets)
            total_loss += loss_val.item()
            # Metric evaluation on real-world scales
            pred_global = de_normalize(predictions, origin_val, theta_val)
            future_global = de_normalize(batch_targets, origin_val, theta_val)

            total_ade += ADE(pred_global, future_global).item()
            total_fde += FDE(pred_global, future_global).item()
    model.train()
    return total_loss / num_batches, total_ade / num_batches, total_fde / num_batches


# resume training logic
start_epoch = 0
best_val_loss = float('inf')

if resume_training and os.path.exists(model_path):
    print(f"Found checkpoint at {model_path}. Loading weights to resume")

    # Load weights safely across devices
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Set this to the next epoch you need to run (you completed up to epoch 4)
    start_epoch = 5
    print(f"Resuming training starting from Epoch {start_epoch}")
print(f"looping starts in {time.time() - start_time:.2f} s")
epochs = 30
loss_list = []
for epoch in range(start_epoch,epochs):
    epoch_start = time.time()
    model.train()
    total_train_loss = 0
    for past,future,_,_ in train_dataloader:
        past, future = past.to(device), future.to(device)
        optimizer.zero_grad()
        pred = model(past)
        loss = loss_func(pred, future)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_dataloader)
    val_loss, val_ade, val_fde = validation_loss(model, val_dataloader, loss_func)

    loss_list.append({"Epoach" : epoch,
                      "Train_Loss": avg_train_loss,
                      "Val_Loss": val_loss,
                      "ADE": val_ade,
                      "FDE": val_fde})
    scheduler.step(val_loss)
    # CheckPoint: Save the absolute best weights based on validation performance
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
    print(f"Epoch: {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | ADE: {val_ade:.2f}m | FDE: {val_fde:.2f}m | Time: {time.time() - epoch_start:.2f}s")

df = pd.DataFrame(loss_list)
df.to_csv("loss_history.csv", index=False)
torch.save(model.state_dict(), model_path)
end_time = time.time()
print(f"Elapsed time: {round((end_time - start_time), 2)} seconds")