# CML_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from CML_dataset import *          # no file_ids needed
from CML_model import CMLMixedRegressor     # your CNN+GRU regression model
from utils import *

def main():
    hepps_set_seeds(114514)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    BATCH_SIZE    = 64
    NUM_EPOCHS    = 50
    LEARNING_RATE = 1e-4

    # 4) Dataset & DataLoader
    dataset = CMLDataset(
        target_length=1024,
        remove_after_resample=10
    )
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True              
    )
    test_loader = DataLoader(
        test_set,  
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True              
    )

    # 5) Model, Loss, Optimizer, Scheduler
    model = CMLMixedRegressor(
        cnn_channels=[16, 32],
        gru_hidden=64,
        gru_layers=1,
        bidirectional=True,
        dropout=0.1,
        output_dim=2   # SBP and DBP
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # 6) Training loop
    train_losses = [] 
    test_losses  = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        count = 0

        for i, (x, sbp, dbp) in enumerate(train_loader, start=1):  # CHANGED: use train_loader
            x   = x.to(device)
            sbp = sbp.to(device)
            dbp = dbp.to(device)
            y   = torch.stack([sbp, dbp], dim=1)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            count += x.size(0)

        train_loss = running_loss / count
        train_losses.append(train_loss)

        # --- validation ---
        model.eval()                  
        val_running = 0.0             
        val_count   = 0               
        with torch.no_grad():         
            for x_val, sbp_val, dbp_val in test_loader:    
                x_val   = x_val.to(device)
                sbp_val = sbp_val.to(device)
                dbp_val = dbp_val.to(device)
                y_val   = torch.stack([sbp_val, dbp_val], dim=1)

                out_val = model(x_val)
                l_val   = criterion(out_val, y_val)
                val_running += l_val.item() * x_val.size(0)
                val_count   += x_val.size(0)

        test_loss = val_running / val_count
        test_losses.append(test_loss)

        # scheduler step on validation loss
        scheduler.step(test_loss)

        tqdm.write(
            f"→ Epoch {epoch}/{NUM_EPOCHS} "
            f"Train Loss: {train_loss:.4f}  "
            f"Test Loss: {test_loss:.4f}"
        )

    print("Training finished.")
    model.eval()

    # 7) Quick inspection on one test batch
    with torch.no_grad():
        x_batch, sbp_batch, dbp_batch = next(iter(test_loader))   
        x_batch = x_batch.to(device)
        preds   = model(x_batch).cpu().numpy()
        true    = np.stack([sbp_batch.numpy(), dbp_batch.numpy()], axis=1)

    for i in range(min(5, len(preds))):
        print(
            f"Sample {i:2d}: "
            f"True SBP/DBP = {true[i,0]:.1f}/{true[i,1]:.1f} | "
            f"Pred = {preds[i,0]:.1f}/{preds[i,1]:.1f}"
        )

    # 8) Plot training & validation loss curves
    epochs = np.arange(1, NUM_EPOCHS+1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train MSE')
    plt.plot(epochs, test_losses,  label='Test MSE')        
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()

    # 10) Plot true vs. predicted SBP & DBP on test batch 
    plt.figure(figsize=(8,4))                             
    idx = np.arange(true.shape[0])                        
    # SBP
    plt.plot(idx, true[:,0], label='True SBP')            
    plt.plot(idx, preds[:,0], linestyle='--', label='Pred SBP') 
    # DBP
    plt.plot(idx, true[:,1], label='True DBP')            
    plt.plot(idx, preds[:,1], linestyle='--', label='Pred DBP') 
    plt.xlabel('Sample index')                            
    plt.ylabel('Blood Pressure (mmHg)')                   
    plt.title('True vs. Predicted SBP & DBP')             
    plt.legend()                                          
    plt.show()                                            

    # 9) Save model
    torch.save(model.state_dict(), 'CML_model.pth')
    print("Model saved as CML_model.pth")

if __name__ == "__main__":
    main()
