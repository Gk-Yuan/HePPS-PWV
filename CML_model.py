import torch
import torch.nn as nn

class CMLMixedRegressor(nn.Module):
    def __init__(self,
                 cnn_channels=[16, 32],
                 gru_hidden=64,
                 gru_layers=1,
                 bidirectional=True,
                 dropout=0.0,
                 output_dim=1):
        super().__init__()
        # === CNN Encoders ===
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels[0], kernel_size=16, padding=2),
            nn.Sigmoid(),
            nn.MaxPool1d(4),
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=16, padding=2),
            nn.Sigmoid(),
            nn.MaxPool1d(4),
            nn.Conv1d(cnn_channels[1], cnn_channels[1], kernel_size=16, padding=2),
            nn.Sigmoid(),
            nn.MaxPool1d(4),
        )
        # === GRU Modules ===
        rnn_input = cnn_channels[1]
        self.gru = nn.GRU(
            rnn_input, gru_hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=bidirectional
        )
        factor = 2 if bidirectional else 1
        # Regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * factor * 2, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, 2)
        wrist = x[:, :, 0].unsqueeze(1)   # (batch, 1, seq_len)
        finger = x[:, :, 1].unsqueeze(1)  # (batch, 1, seq_len)

        w_enc = self.cnn(wrist)     # (batch, 32, seq_len/64)
        f_enc = self.cnn(finger)   # (batch, 32, seq_len/32)

        w_seq = w_enc.permute(0, 2, 1)        # (batch, time_w, feat)
        f_seq = f_enc.permute(0, 2, 1)        # (batch, time_f, feat)

        w_out, _ = self.gru(w_seq)      # (batch, time_w, hidden*factor)
        f_out, _ = self.gru(f_seq)     # (batch, time_f, hidden*factor)

        # take final timestep
        w_last = w_out[:, -1, :]
        f_last = f_out[:, -1, :]

        combined = torch.cat([w_last, f_last], dim=1)
        return self.regressor(combined)