import torch.nn as nn

import torch
import torch.nn as nn


class VoltammogramConvNet(nn.Module):
    def __init__(self, input_length, num_outputs=3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.Tanh(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, input_length]
        return self.model(x)

class VoltammogramLSTMNet(nn.Module):
    def __init__(self, input_length, num_outputs=3, hidden_dim=128, num_layers=2):
        super().__init__()
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=1,          # because input is 1D signal
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )

        self.norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_outputs)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]
        normed = self.norm(last_timestep)
        return self.fc(normed)

