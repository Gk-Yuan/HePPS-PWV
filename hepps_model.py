import torch
import torch.nn as nn

class HePPSNet(nn.Module):
    def __init__(self,
                 cnn_channels=[16, 32],
                 gru_hidden=64,
                 gru_layers=1,
                 bidirectional=True,
                 dropout=0,
                 num_classes=5):
        super().__init__()
        # === CNN Encoders ===
        # Each encoder processes a single-channel input of shape (batch, 1, seq_len)
        # Wrist encoder reduces sequence length by 4 -> 4 -> 2 (total factor 32)
        self.encoder_wrist = nn.Sequential(
            # 1st conv layer
            nn.Conv1d(1, cnn_channels[0], kernel_size=16, stride=1, padding=2), # (batch, 16, seq)
            nn.Sigmoid(),
            nn.AvgPool1d(4),                                                    # (batch, 16, seq/4)
            # 2nd conv layer
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=16, stride=1, padding=2),   # (batch, 32, seq/4)
            nn.Sigmoid(),
            nn.AvgPool1d(4),                                                    # (batch, 32, seq/16)
            # 3rd conv layer
            nn.Conv1d(cnn_channels[1], cnn_channels[1], kernel_size=16, stride=1, padding=2),  # (batch, 32, seq/16)
            nn.Sigmoid(),
            nn.AvgPool1d(2),                                                    # (batch, 32, seq/32)
        )
        # Finger encoder reduces sequence length by 3 -> 3 (total factor 9)
        self.encoder_finger = nn.Sequential(
            # 1st conv layer
            nn.Conv1d(1, cnn_channels[0], kernel_size=16, stride=1, padding=2), # (batch, 16, seq)
            nn.Sigmoid(),
            nn.AvgPool1d(4),                                                    # (batch, 16, seq/4)
            # 2nd conv layer
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=16, stride=1, padding=2),   # (batch, 32, seq/4)
            nn.Sigmoid(),
            nn.AvgPool1d(4),                                                    # (batch, 32, seq/16)
            # 3rd conv layer
            nn.Conv1d(cnn_channels[1], cnn_channels[1], kernel_size=16, stride=1, padding=2),  # (batch, 32, seq/16)
            nn.Sigmoid(),
            nn.AvgPool1d(2),                                                    # (batch, 32, seq/32)
        )
        # === GRU Modules ===
        # Input to each GRU: sequence of length seq_len/32 for wrist and seq_len/32 for finger,
        # with feature dimension = cnn_channels[1]
        rnn_input = cnn_channels[1]
        self.gru_wrist = nn.GRU(
            rnn_input,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.gru_finger = nn.GRU(
            rnn_input,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        factor = 2 if bidirectional else 1
        # Classifier input dim: wrist_hidden*factor + finger_hidden*factor
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * factor * 2, num_classes),    # -> (batch, num_classes)
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x, length):
        # x: (batch, seq_len, 2)  -- channels last
        # Split channels
        wrist = x[:, :, 0].unsqueeze(1)   # -> (batch, 1, seq_len)
        finger = x[:, :, 1].unsqueeze(1)  # -> (batch, 1, seq_len)

        # Encode each channel
        w_enc = self.encoder_wrist(wrist)   # -> (batch, 32, seq_len/32)
        f_enc = self.encoder_finger(finger) # -> (batch, 32, seq_len/32)

        # Prepare for RNN: (batch, time, features)
        w_seq = w_enc.permute(0, 2, 1)  # -> (batch, seq_len/32, 32)
        f_seq = f_enc.permute(0, 2, 1)  # -> (batch, seq_len/32, 32)

        # GRU processing
        w_out, _ = self.gru_wrist(w_seq)   # -> (batch, seq_len/32, hidden*factor)
        f_out, _ = self.gru_finger(f_seq)  # -> (batch, seq_len/32, hidden*factor)

        # Take the final time-step
        w_last = w_out[:, -1, :]  # -> (batch, hidden*factor)
        f_last = f_out[:, -1, :]  # -> (batch, hidden*factor)

        # Concatenate wrist + finger features
        combined = torch.cat((w_last, f_last), dim=1)  # -> (batch, hidden*factor*2)

        merged = torch.cat([combined, length], dim=1)     # (batch, hidden*factor*2 + 1)
        logits = self.classifier(merged)                  # (batch, num_classes)
        return logits
