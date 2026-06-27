import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loader import ArgoDataset  # Uses your tensor dataset class
from model import LSTMPredictor


def de_normalize(pred_normalized, origin, theta):
    target_device = pred_normalized.device
    theta = theta.to(target_device)
    cos_t = torch.cos(theta).to(target_device)
    sin_t = torch.sin(theta).to(target_device)
    origin = origin.to(target_device)

    x_rotated = pred_normalized[..., 0] * cos_t.unsqueeze(1) - pred_normalized[..., 1] * sin_t.unsqueeze(1)
    y_rotated = pred_normalized[..., 0] * sin_t.unsqueeze(1) + pred_normalized[..., 1] * cos_t.unsqueeze(1)

    x_global = x_rotated + origin[..., 0].unsqueeze(1)
    y_global = y_rotated + origin[..., 1].unsqueeze(1)

    return torch.stack([x_global, y_global], dim=-1)


def visualize_predictions(model, dataloader, device, num_samples=3):
    model.eval()
    samples_plotted = 0

    # Create an output directory for plots
    os.makedirs("../plots", exist_ok=True)

    with torch.no_grad():
        for past, future, origin, theta in dataloader:
            if samples_plotted >= num_samples:
                break

            # Move only what the model needs to GPU
            past_gpu = past.to(device)
            preds_normalized = model(past_gpu)

            # Map predictions and ground truth back to global coordinates
            preds_global = de_normalize(preds_normalized, origin, theta).cpu().numpy()
            past_global = de_normalize(past[..., :2], origin, theta).cpu().numpy()  # Only take X,Y positions
            future_global = de_normalize(future, origin, theta).cpu().numpy()

            # Plot individual samples from the batch
            for i in range(past.size(0)):
                if samples_plotted >= num_samples:
                    break

                plt.figure(figsize=(10, 6))

                # Plot History (Past)
                plt.plot(past_global[i, :, 0], past_global[i, :, 1], color='blue', marker='o',
                         label='Past History (2s)', markersize=4)
                # Mark the origin point (t = 0)
                plt.plot(origin[i, 0].item(), origin[i, 1].item(), color='black', marker='X', markersize=10,
                         label='Current Position')

                # Plot Ground Truth Future
                plt.plot(future_global[i, :, 0], future_global[i, :, 1], color='green', marker='s',
                         label='Ground Truth Future (3s)', markersize=4)

                # Plot LSTM Prediction
                plt.plot(preds_global[i, :, 0], preds_global[i, :, 1], color='red', linestyle='--', marker='^',
                         label='LSTM Prediction', markersize=4)

                plt.title(f"Trajectory Prediction Sample {samples_plotted + 1}")
                plt.xlabel("Global X Coordinate (meters)")
                plt.ylabel("Global Y Coordinate (meters)")
                plt.legend()
                plt.grid(True)
                plt.axis('equal')  # Keep aspect ratio 1:1 to see true turning geometry

                # Save plot to files
                plot_path = f"../plots/sample_{samples_plotted + 1}.png"
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()

                print(f"🖼️ Saved visualization plot to: {plot_path}")
                samples_plotted += 1


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "../models/model.pth"
    test_folder = "/content/tensor_data/val"

    # Load dataset and model
    print("Loading validation dataset for evaluation...")
    test_dataset = ArgoDataset(test_folder)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)  # Shuffle to get random behaviors

    print("Loading trained LSTM weights...")
    model = LSTMPredictor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Run visualization
    print("Generating trajectory plots...")
    visualize_predictions(model, test_dataloader, device, num_samples=10)
    print("Testing and visualization cycle complete!")