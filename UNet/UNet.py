'''

    Modulo per implementazione di U-Net e U-Net con soft attention gates.
    Questi moduli sono utilizzati per segmentazione di keypoints tramite heatmap.
    
'''


import os
import torch
import torch.nn as nn
from torch import nn, optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# ---------------------------------------------------------------------------------------- #




'''
    Modulo per implementazione di U-Net.
    
    U-Net è una rete neurale convoluzionale progettata per segmentazione tramite heatmap.
    È composta da più encoder (che riducono la dimensione dell'immagine) e più decoder (che ricostruiscono l'immagine).
    La struttura a U permette di mantenere le informazioni spaziali grazie alle connessioni skip tra encoder e decoder.
    Ogni blocco di convoluzione (DoubleConv) consiste in due strati di convoluzione seguiti da ReLU e Batch Normalization.
'''




class DoubleConv(nn.Module):
    """
    Blocco di due convoluzioni + ReLU, utilizzato sia negli encoder che nei decoder di U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            # convoluzione 3x3, padding=1 è un bordo di pixel 0 per mantenere le dimensioni dell'immagine
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  
            nn.BatchNorm2d(out_channels), # normalizzazione batch per stabilizzare l'addestramento
            nn.ReLU(inplace=True), # attivazione ReLU
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)





class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        """
        U-Net con immagini di input 800x800

        Parameters
        ----------
        in_channels : int
            Numero di canali in input (es. 3 per immagini RGB).
        out_channels : int
            Numero di canali in output (es. 1 per heatmap o mask binaria).
        """
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)     # (B, 64, 800, 800)
        x2 = self.down2(self.pool1(x1))  # (B, 128, 400, 400)
        x3 = self.down3(self.pool2(x2))  # (B, 256, 200, 200)
        x4 = self.down4(self.pool3(x3))  # (B, 512, 100, 100)

        # Bottleneck
        x5 = self.bottleneck(self.pool4(x4))  # (B, 1024, 50, 50)

        # Decoder
        x = self.up4(x5)                      # (B, 512, 100, 100)
        x = self.dec4(torch.cat([x, x4], dim=1))  # Skip connection (da encoder a decoder)

        x = self.up3(x)                       # (B, 256, 200, 200)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)                       # (B, 128, 400, 400)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)                       # (B, 64, 800, 800)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.out_conv(x)              # (B, out_channels, 800, 800)
    
    
    
    
# ---------------------------------------------------------------------------------------- #




'''
    Modulo per implementazione di UMNet aventi soft attention gates.
    
    Questi moduli sono utilizzati per migliorare la capacità del modello di focalizzarsi
    su regioni rilevanti dell'immagine durante il processo di decodifica.
    Gli Attention Gates (AG) sono utilizzati per pesare le caratteristiche dell'encoder
    in base alla loro rilevanza per le caratteristiche del decoder.
'''



class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi




class UNetWithAttention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        # Attention Gates
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionGate(F_g=64,  F_l=64,  F_int=32)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))

        # Decoder + Attention
        d4 = self.up4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = self.decoder4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = self.decoder3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = self.decoder2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = self.decoder1(torch.cat([d1, e1_att], dim=1))

        return self.final(d1)
    



# ---------------------------------------------------------------------------------------- #




'''
    Funzioni per addestrare il modello U-Net.
'''




class HeatmapDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_size=(800, 800)):
        '''
        Dataset per caricare immagini e heatmap (o maschere) per l'addestramento di U-Net.
        Parameters
        ----------
        image_dir : str
            Directory contenente le immagini.
        label_dir : str
            Directory contenente le heatmap corrispondenti.
        image_size : tuple
            Dimensione a cui ridimensionare le immagini (default: (800, 800)).
        '''
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size

        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')]) # lista ordinata dei file .png

        self.transform = transforms.Compose([
            transforms.ToTensor() # Converti le immagini in tensori con valori tra 0 e 1
        ])
        
        
        

    def __len__(self):
        '''
        Restituisce il numero totale di esempi nel dataset (numero di immagini).
        '''
        return len(self.image_filenames)




    def __getitem__(self, idx):
        '''
        Carica una coppia (immagine, heatmap) all'indice idx
        Parameters
        ----------
        idx : int
            Indice dell'esempio da caricare.
        Returns
        -------
        tuple
            Una tupla contenente l'immagine e la heatmap (o maschera) come tensori PyTorch.
        '''
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.image_filenames[idx])

        image = Image.open(img_path).convert("L") # Converti in scala di grigi
        label = Image.open(label_path).convert("L") # Converti in scala di grigi

        image = image.resize(self.image_size) # Ridimensiona l'immagine
        label = label.resize(self.image_size) # Ridimensiona la heatmap

        image = self.transform(image) # Converti in tensore
        label = self.transform(label) # Converti in tensore

        return image, label




def get_dataloaders(data_dir, batch_size=8, image_size=(800, 800)):
    '''
    Crea i DataLoader per il training e la validazione del modello U-Net.
    Un dataloader è un oggetto che permette di iterare sui dati in batch (comodo per multithreading).
    Parameters
    ----------
    data_dir : str
        Directory principale contenente le sottodirectory 'images' e 'labels'.
    batch_size : int
        Numero di esempi per batch (default: 8).
    image_size : tuple
        Dimensione a cui ridimensionare le immagini (default: (800, 800)).
    Returns
    -------
    tuple
        Due DataLoader: uno per il training e uno per la validazione.
    '''
    train_dataset = HeatmapDataset(
        image_dir=os.path.join(data_dir, 'images', 'train'),
        label_dir=os.path.join(data_dir, 'labels', 'train'),
        image_size=image_size
    )

    val_dataset = HeatmapDataset(
        image_dir=os.path.join(data_dir, 'images', 'val'),
        label_dir=os.path.join(data_dir, 'labels', 'val'),
        image_size=image_size
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader




def train_unet(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device='cuda', patience=5):
    '''
    Funzione per addestrare il modello U-Net.
    Parameters
    ----------
    model : nn.Module
        Il modello U-Net da addestrare.
    train_loader : DataLoader
        DataLoader per il training.
    val_loader : DataLoader
        DataLoader per la validazione.
    num_epochs : int
        Numero di epoche per l'addestramento (default: 20).
    lr : float
        Learning rate per l'ottimizzatore Adam (default: 1e-3).
    device : str
        Dispositivo su cui eseguire l'addestramento ('cuda' per GPU, 'cpu' per CPU).
    Returns
    -------
    None:
        Addestra il modello e salva il miglior modello basato sulla loss.
    '''
    
    '''
    Alcune note:
    - Adam (Adaptive Moment Estimation) è un ottimizzatore, cioè aggiorna i pesi del modello per minimizzare la loss.
    - MSE (Mean Squared Error) è una funzione di loss che misura la differenza quadratica media tra le predizioni del modello e i valori reali.
    - La loss viene calcolata come la media della MSE su tutti i batch del dataset.
    - La validazione viene eseguita alla fine di ogni epoca per monitorare le prestazioni del modello su dati non visti.
    - Il modello viene salvato se la loss di validazione migliora rispetto alla migliore loss precedente.
    - tqdm è una libreria che fornisce una barra di avanzamento per i loop, utile per monitorare l'addestramento.
    - pos_weight = se la rete commette un errore quando prevede un pixel che dovrebbe essere positivo (cioè, un pixel del centro 
      del cerchio nella tua heatmap), quel costo dell'errore viene moltiplicato per il valore di pos_weight.
      Il suo scopo principale è quello di affrontare il problema dello squilibrio di classi (class imbalance), 
      dove i pixel di "sfondo" (negativi) sono numericamente molto più abbondanti dei pixel di "interesse" (positivi).
      Qui è calcolata come "area dei pixel negativi / somma totale delle intensità dei pixel positivi".
    '''
    
    # Imposta il dispositivo (GPU o CPU)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([150], device=device)) 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # --------- TRAIN ---------
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

            del images, labels, outputs, loss
            torch.cuda.empty_cache()

        train_loss /= len(train_loader.dataset)

        # --------- VALIDATION ---------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                del images, labels, outputs, loss
                torch.cuda.empty_cache()

        val_loss /= len(val_loader.dataset)

        # --------- LOG ---------
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"[GPU] alloc: {torch.cuda.memory_allocated() / 1024**2:.1f} MB | max: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

        # --------- EARLY STOPPING & SAVE ---------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_unet.pth")
            print("Nuovo modello salvato (val loss migliorata)")
        else:
            patience_counter += 1
            print(f"Nessun miglioramento. patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping attivato.")
                break



# ---------------------------------------------------------------------------------------- #




'''
    Funzioni per estrarre coordinate discrete di keypoints da una heatmap.
    Utile per calcolare metriche come PCK (Percentage of Correct Keypoints), precision, recall.    
'''


def extract_keypoints_from_heatmap(heatmap, threshold=0.5):
    """
    Estrae le coordinate dei keypoints da una heatmap.
    
    Parameters
    ----------
    heatmap : torch.Tensor
        Heatmap con valori tra 0 e 1.
    threshold : float
        Soglia per considerare un pixel come keypoint (default: 0.5).
    
    Returns
    -------
    list
        Lista con le coordinate dei keypoints e le loro covarianze.
    """
    # Converte la heatmap da tensore PyTorch a numpy array 2D rimuovendo dimensioni inutili
    # Conversione sicura: torch tensor → numpy, altrimenti lascia com'è
    if hasattr(heatmap, 'cpu'):
        heatmap = heatmap.squeeze().cpu().numpy()
    else:
        heatmap = np.squeeze(heatmap)    
    
    # Binarizza la heatmap con la soglia, pixel > threshold diventano 1, altrimenti 0
    binary = (heatmap > threshold).astype(np.uint8)
    
    # Etichetta i gruppi connessi (componenti connesse) nella matrice binaria
    labeled, num_features = scipy.ndimage.label(binary)

    keypoints = []
    # Ciclo su ciascun gruppo connesso trovato (ogni potenziale keypoint)
    for i in range(1, num_features + 1):
        # Crea una maschera booleana per selezionare il gruppo i-esimo
        mask = labeled == i
        
        # Pesa la heatmap originale con la maschera per considerare solo quel gruppo
        weights = heatmap * mask
        
        # Somma totale dei pesi all'interno del gruppo
        total_weight = np.sum(weights)
        if total_weight == 0:
            # Se il peso totale è zero (gruppo nullo) passa al successivo
            continue

        # Costruisce coordinate x,y per ogni pixel della heatmap
        x_coords, y_coords = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]))
        
        # Moltiplica le coordinate x e y per i pesi e appiattisce l'array
        x = (x_coords * weights).ravel() # .ravel = appiattisce l'array 2D in 1D
        y = (y_coords * weights).ravel()
        
        # Unisce i vettori x,y e normalizza per la somma dei pesi per ottenere la media ponderata
        points = np.vstack((x, y)) / total_weight

        # Calcola la media (coordinate centrali) del keypoint: mu_x e mu_y
        mu = np.sum(points, axis=1)

        # Calcola la differenza delle coordinate dai valori medi
        diffs = np.vstack((x_coords.ravel() - mu[0], y_coords.ravel() - mu[1]))
        
        # Calcola la matrice di covarianza pesata, che indica la "forma" e distribuzione spaziale del keypoint
        cov = np.cov(diffs, aweights=weights.ravel())

        # Aggiunge alla lista il keypoint con coordinate medie e covarianza
        keypoints.append((mu[0], mu[1], cov))  # x, y, matrice di covarianza

    return keypoints




def plot_keypoints_with_covariances(keypoints):
    """
    Plotta i keypoints su due grafici affiancati:
    - A sinistra solo i punti (x, y)
    - A destra i punti con le ellissi di covarianza

    Parameters
    ----------
    keypoints : list of tuples
        Lista di tuple (x, y, cov) dove cov è la matrice di covarianza 2x2
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Primo grafico: punti semplici
    axs[0].set_title("Keypoints (punti)")
    axs[0].set_aspect('equal')
    axs[0].grid(True)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    # Secondo grafico: punti + ellissi covarianza
    axs[1].set_title("Keypoints con ellissi di covarianza")
    axs[1].set_aspect('equal')
    axs[1].grid(True)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')

    for (x, y, cov) in keypoints:
        # Primo plot: solo il punto
        axs[0].plot(x, y, 'ro')

        # Secondo plot: punto
        axs[1].plot(x, y, 'ro')

        # Calcolo degli autovalori e autovettori della covarianza
        vals, vecs = np.linalg.eigh(cov)

        # Ordinamento degli autovalori dal più grande al più piccolo
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # Angolo di rotazione dell'ellisse (in gradi)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        # Ampiezza degli assi dell'ellisse = 2 * sqrt(autovalori)
        width, height = 2 * np.sqrt(vals)

        # Crea l'ellisse e la aggiunge al grafico
        ell = Ellipse(xy=(x, y), width=width, height=height, angle=angle,
                      edgecolor='blue', fc='none', lw=2, alpha=0.6)
        axs[1].add_patch(ell)

    # Limiti automatici con padding
    for ax in axs:
        all_x = [kp[0] for kp in keypoints]
        all_y = [kp[1] for kp in keypoints]
        ax.set_xlim(min(all_x) - 5, max(all_x) + 5)
        ax.set_ylim(min(all_y) - 5, max(all_y) + 5)

    plt.tight_layout()
    plt.show()
