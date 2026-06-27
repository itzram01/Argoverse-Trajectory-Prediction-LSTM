import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=256, num_layers= 2,batch_first=True,dropout=0.2)
        self.fc = nn.Linear(256, 60)
    def forward(self, x):
        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x)
        # Extract the hidden features from the very last sequence step
        last_step_features = out[:, -1, :]
        output = self.fc(last_step_features)
        return output.view(-1, 30, 2)