import sys
import torch

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # aggiorna subito il file

    def flush(self):
        pass  # richiesto per compatibilità con Python

sys.stdout = Logger("log.txt")
sys.stderr = sys.stdout  # gli errori finiscono nel log

# ----  codice ----
sys.path.append(r'/user/gr1/delphi/dghezzi/UNet')

from UNet import get_dataloaders, train_unet, UNetWithAttention, pos_weight_calculator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training su dispositivo: {device}")
torch.cuda.empty_cache()

model = UNetWithAttention(in_channels=1, out_channels=1)

datapath = '/user/gr1/delphi/dghezzi/SharedDataset_22500_2500_150_175_npy/UNet/800px'
pos_weight_= pos_weight_calculator(datapath)

train_loader, val_loader = get_dataloaders(data_dir=datapath, batch_size=8, image_size=(800,800))

train_unet(model, train_loader, val_loader, num_epochs=100, lr=1e-3,
           device=device, patience=7, f1_earlystopping=False,
           pos_weight=pos_weight_, img_size=800)

print("Training completato e cache GPU svuotata.")
