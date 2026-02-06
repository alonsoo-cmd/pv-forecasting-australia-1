import torch
import torch.nn as nn


class LSTM_FCN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_window,
        dropout,
    ):
        super().__init__()

        # -------- LSTM branch --------
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # -------- FCN (Fully Convolutional Network) branch --------
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=8, padding=4)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # -------- Final Output Head --------
        # Concatenates outputs from both branches (hidden_size + 128)
        self.fc = nn.Linear(hidden_size + 128, output_window)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        # ---- LSTM path ----
        lstm_out, _ = self.lstm(x)
        lstm_feat = lstm_out[:, -1, :]   # Extract last timestep

        # ---- FCN path ----
        y = x.transpose(1, 2)            # Reshape to (batch, input_size, seq_len) for Conv1d
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.relu(self.bn3(self.conv3(y)))
        fcn_feat = y.mean(dim=2)         # Global Average Pooling

        # ---- Merge Branches ----
        features = torch.cat([lstm_feat, fcn_feat], dim=1)
        features = self.dropout(features)

        return self.fc(features)