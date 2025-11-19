import torch
import torch.nn as nn

class CMLMixedRegressor(nn.Module):
    """
    Inputs
    ------
    x       : (B, T, 2)            # wrist + finger waveforms (current)
    feat24  : (B, 24)              # 24 first-order handcrafted features (current)
    meta4   : (B, 4)               # [height, weight, SBP_ref, DBP_ref] (standardized if possible)

    Behavior
    --------
    - CNN→GRU encodes wrist/finger separately, take last states, concat -> (B, 2H*)
    - LayerNorm applied to embeddings, features, and meta (standardize-everything treatment)
    - Regressor consumes [embed, feat24, meta4] and outputs:
        * if predict_delta=False: y_hat = regressor(...)
        * if predict_delta=True : y_hat = y_ref + regressor(...), where y_ref comes from meta4[:, 2:4]
    """

    def __init__(self,
                 cnn_channels=[32, 64],
                 gru_hidden=64,
                 gru_layers=1,
                 bidirectional=False,
                 dropout=0.0,
                 output_dim=2,                # predict [SBP, DBP] by default
                 feat_dim=24,
                 meta_dim=4,                  # [height, weight, SBP_ref, DBP_ref]
                 predict_delta=True):         # add Δ to y_ref (calibration form)
        super().__init__()

        self.predict_delta = predict_delta
        self.feat_dim = feat_dim
        self.meta_dim = meta_dim

        # === CNN encoder (shared for wrist/finger) ===
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels[0], kernel_size=16, padding=2),
            nn.Sigmoid(),
            nn.MaxPool1d(2),

            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=16, padding=2),
            nn.Sigmoid(),
            nn.MaxPool1d(2),

            nn.Conv1d(cnn_channels[1], cnn_channels[1], kernel_size=16, padding=2),
            nn.Sigmoid(),
            nn.MaxPool1d(2),
        )

        # === GRU encoder ===
        rnn_input = cnn_channels[1]
        self.gru = nn.GRU(
            input_size=rnn_input,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        factor = 2 if bidirectional else 1

        # === LayerNorm blocks (standardize everything inside) ===
        self.ln_embed = nn.LayerNorm(gru_hidden * factor)  # applied to each last state
        self.ln_feat  = nn.LayerNorm(feat_dim)             # 24-D features
        self.ln_meta  = nn.LayerNorm(meta_dim)             # [h, w, SBP_ref, DBP_ref]

        # === Final regressor head ===
        reg_in_dim = (gru_hidden * factor * 2) + feat_dim + meta_dim  # [w_last, f_last] + 24 + 4
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(reg_in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)   # predicts [SBP, DBP] or Δ[SBP, DBP]
        )

    def _encode_channel(self, one_channel_wave: torch.Tensor) -> torch.Tensor:
        """
        one_channel_wave: (B, 1, T)
        returns: last GRU state (B, H*)
        """
        h = self.cnn(one_channel_wave)      # (B, C, T')
        h = h.permute(0, 2, 1)              # (B, T', C)
        out, _ = self.gru(h)                # (B, T', H*)
        last = out[:, -1, :]                # (B, H*)
        return self.ln_embed(last)          # LayerNorm on embedding

    def forward(self, x, feat24, meta4):
        """
        x     : (B, T, 2)
        feat24: (B, 24)
        meta4 : (B, 4) = [height, weight, SBP_ref, DBP_ref]
        """
        # Split channels: wrist & finger
        wrist  = x[:, :, 0].unsqueeze(1)     # (B, 1, T)
        finger = x[:, :, 1].unsqueeze(1)     # (B, 1, T)

        # Encode each channel with shared CNN→GRU and standardize
        w_last = self._encode_channel(wrist)   # (B, H*)
        f_last = self._encode_channel(finger)  # (B, H*)

        # Normalize feature blocks
        feat24 = self.ln_feat(feat24)          # (B, 24)
        meta4  = self.ln_meta(meta4)           # (B, 4) standardized internally

        # Concatenate: embeddings + features + meta
        embed = torch.cat([w_last, f_last], dim=1)  # (B, 2H*)
        z = torch.cat([embed, feat24, meta4], dim=1)

        # Regress
        y_head = self.regressor(z)             # (B, output_dim)

        # If calibration-delta mode: add y_ref back
        if self.predict_delta:
            # y_ref expected in meta4's last two positions (after LayerNorm it's still aligned)
            y_ref = meta4[:, -2:]              # (B, 2)
            # NOTE: If you z-scored meta before model, pass unnormalized y_ref separately instead.
            # In that case, change meta_dim and feed y_ref as a separate arg.
            return y_ref + y_head

        return y_head
