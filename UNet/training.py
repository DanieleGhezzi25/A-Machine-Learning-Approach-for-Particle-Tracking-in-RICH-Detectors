'''

    Script da eseguire da terminale in "screen" per il training della UNet.
    Lo screen è utile per evitare che il training venga interrotto in caso di disconnessione.
    Per avviare lo screen: screen -S training
    Per uscire dallo screen: Ctrl+A, D
    Per rientrare nello screen: screen -r training
    Per terminare lo screen: exit

'''


import torch
torch.cuda.empty_cache()

from torch.utils.data import DataLoader
from UNet import UNet, get_dataloaders, train_unet, UNetWithAttention

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training su dispositivo: {device}")

# Istanzia modello
model = UNetWithAttention(in_channels=1, out_channels=1)  # o 3 se RGB

# Carica i dataloader (adatta il path)
datapath = '/user/gr1/delphi/dghezzi/UNet/UNet_dataset/92000_8000_160_180_npy'
train_loader, val_loader = get_dataloaders(data_dir=datapath, batch_size=8, image_size=(800,800))

# Avvia il training
train_unet(model, train_loader, val_loader, num_epochs=100, lr=1e-3, device=device, patience=7)

torch.cuda.empty_cache()
print("Training completato e cache GPU svuotata.")