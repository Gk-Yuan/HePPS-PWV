# hepps_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from tqdm import tqdm

from hepps_dataset import HePPSDataset
from hepps_model   import HePPSNet

def main():
    # 1) Device detection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 2) Hyperparameters
    BATCH_SIZE    = 128
    NUM_EPOCHS    = 10
    LEARNING_RATE = 1e-4

    # 3) Dataset + DataLoader
    file_ids = [
        "rest",
        "exercise",
        "deep_breath",
        "hold_breath_1",
        "hold_breath_2",
        "caffeine"
    ]

    dataset = HePPSDataset(
        file_identifiers=file_ids,
        target_length=512,
        remove_after_resample=10
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 4) Label map & normalization helper
    base_map = {
        "rest": 0,
        "exercise": 1,
        "deep_breath": 2,
        "hold_breath": 3,
        "caffeine": 4
    }

    # 5) Model, Loss, Optimizer, Scheduler, AMP scaler
    model = HePPSNet(
        cnn_channels=[16, 32],
        gru_hidden=64,
        gru_layers=1,
        bidirectional=False,
        dropout=0.0,
        num_classes=5
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        # weight_decay=1e-5
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )


    # def normalize_label(fid):
    #     parts = fid.rsplit('_',1)
    #     return parts[0] if len(parts)==2 and parts[1].isdigit() else fid

    # 6) Training loop
    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        model.train()
        total_loss = 0.0
        correct = 0

        total   = 0

        for i, (x, lengths, labels) in enumerate(loader, start=1):
            x = x.to(device)
            y = torch.tensor([base_map[l] for l in labels ],
                            device=device, 
                            dtype=torch.long)

            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # accumulate for reporting
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += x.size(0)

            if i % 10 == 0:
                avg_loss = total_loss / total
                avg_acc  = 100 * correct / total
                tqdm.write(
                    f"Epoch {epoch}/{NUM_EPOCHS} | "
                    f"Batch {i}/{len(loader)} | "
                    f"Loss {avg_loss:.4f} | Acc {avg_acc:5.2f}%"
                )

        # end-of-epoch summary
        epoch_loss = total_loss / total
        epoch_acc  = 100 * correct / total
        tqdm.write(f"→ Epoch {epoch} done: Loss {epoch_loss:.4f}, Acc {epoch_acc:5.2f}%\n")
        scheduler.step(epoch_loss)

    print("Training complete.")
    model.eval()
    with torch.no_grad():
        # grab one batch
        x_batch, labels = next(iter(loader))
        x_batch = x_batch.to(device)
        y_true = [base_map[l] for l in labels]
        logits = model(x_batch)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1).cpu().tolist()

    # print them side by side
    for i, (t, p, pr) in enumerate(zip(labels, preds, probs.cpu().tolist())):
        print(f"  Sample {i:2d} | true: {y_true:15s} | pred: {p:<2d} | probs: {pr}")

if __name__ == "__main__":
    main()
