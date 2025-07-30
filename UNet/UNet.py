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
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
import cv2
from scipy.optimize import linear_sum_assignment
import time



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
    Modulo per implementazione di UNet aventi soft attention gates.
    
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

        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.npy')]) # lista ordinata dei file .npy

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

        fname = self.image_filenames[idx]

        image = np.load(os.path.join(self.image_dir, fname))  # H x W
        label = np.load(os.path.join(self.label_dir, fname))  # H x W

        # Espandi a 1 canale (1 x H x W)
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        # if self.transform:
        #     image, label = self.transform(image, label)

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





class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # sigmoid qui dentro, perché usi BCEWithLogits fuori
        preds = preds.contiguous().view(preds.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            preds.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        return 1 - dice.mean()





def train_unet(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device='cuda', patience=5, binary_threshold=0.97):
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
    - La loss è la BCEWithLogitsLoss
    - La loss viene calcolata come la media della BCEWithLogitsLoss su tutti i batch del dataset.
    - La validazione viene eseguita alla fine di ogni epoca per monitorare le prestazioni del modello su dati non visti.
    - Il modello viene salvato se la loss di validazione migliora rispetto alla migliore loss precedente.
    - tqdm è una libreria che fornisce una barra di avanzamento per i loop, utile per monitorare l'addestramento.
    - pos_weight = se la rete commette un errore quando prevede un pixel che dovrebbe essere positivo (cioè, un pixel del centro 
      del cerchio nella heatmap), quel costo dell'errore viene moltiplicato per il valore di pos_weight.
      Il suo scopo principale è quello di affrontare il problema dello squilibrio di classi (class imbalance), 
      dove i pixel di "sfondo" (negativi) sono numericamente molto più abbondanti dei pixel di "interesse" (positivi).
      Qui è calcolata come "area dei pixel negativi / somma totale delle intensità dei pixel positivi" (è stata fatta una stima con area gaussiana 2pi*sigma**2).
    '''
    
    # Imposta il dispositivo (GPU o CPU)
    model = model.to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([275], device=device)) 
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    best_val_loss = float('inf')
    best_val_loss_1 = float('inf')
    best_f1 = 0.0
    best_f1_1 = 0.0
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
                loss = 0.5 * bce_loss(outputs, labels) + 0.5 * dice_loss(outputs, labels)

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
        f1_scores = []
        train_bce_total = 0.0
        train_dice_total = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast():
                    outputs = model(images)
                    bce = bce_loss(outputs, labels)
                    dice = dice_loss(outputs, labels)
                    loss = 0.5 * bce + 0.5 * dice

                train_bce_total += bce.item() * images.size(0)
                train_dice_total += dice.item() * images.size(0)
                val_loss += loss.item() * images.size(0)

                # Calcolo F1 per ciascuna immagine nel batch
                for i in range(images.size(0)):
                    outputs[i] = torch.sigmoid(outputs[i])  # Applica sigmoid (da NON fare prima con BCEWithLogitsLoss !!!)
                    thr = outputs[i].max().item() * binary_threshold
                    pred_kpts = extract_predicted_keypoints(outputs[i].cpu(), threshold=thr)
                    gt_kpts = extract_predicted_keypoints(labels[i].cpu(), threshold=thr)
                    '''
                    if i == 0:
                        print(f'outputs[i].shape, outputs[i].max(), outputs[i].min(), outputs[i].mean() # DEBUG')
                        print(outputs[i].shape, outputs[i].max(), outputs[i].min(), outputs[i].mean()) # DEBUG
                        print(f'labels[i].shape, labels[i].max(), labels[i].min(), labels[i].mean()) # DEBUG')
                        print(labels[i].shape, labels[i].max(), labels[i].min(), labels[i].mean()) # DEBUG
                    '''
                    p, r, f1 = compute_pck_metrics(gt_kpts, pred_kpts, thresholds=[2, 4, 6])
                    f1_scores.append(f1)

                del images, labels, outputs, loss
                torch.cuda.empty_cache()

        val_bce_avg = train_bce_total / len(val_loader.dataset)
        val_dice_avg = train_dice_total / len(val_loader.dataset)
        f1_scores = np.array(f1_scores)
        val_loss /= len(val_loader.dataset)
        mean_f1 = np.mean(f1_scores, axis=0)
        weighted_f1 = 0.2 * mean_f1[0] + 0.5 * mean_f1[1] + 0.3 * mean_f1[2]

        # --------- LOG ---------
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"Val BCE Loss: {val_bce_avg:.6f} | Val Dice Loss: {val_dice_avg:.6f}")
        print(f"F1 (thresholds = 2px): {mean_f1[0]:.4f} | F1 (thresholds = 4px): {mean_f1[1]:.4f} | F1 (thresholds = 6px): {mean_f1[2]:.4f}")
        print(f"Weighted F1: 0.2 x {mean_f1[0]:.4f} + 0.5 x {mean_f1[1]:.4f} + 0.3 x {mean_f1[2]:.4f} = {weighted_f1:.4f}")
        print(f"[GPU] alloc: {torch.cuda.memory_allocated() / 1024**2:.1f} MB | max: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

        # --------- EARLY STOPPING & SAVE ---------
        if val_loss <= best_val_loss and weighted_f1 >= best_f1:
            best_val_loss = val_loss
            best_f1 = weighted_f1
            best_val_loss_1 = val_loss
            best_f1_1 = weighted_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_unet.pth")
            print(" ==> Nuovo modello salvato (val loss e F1 migliorati)")
        elif val_loss <= best_val_loss_1:
            best_val_loss_1 = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_unet_for_val_loss.pth")
            print(" ==> Nuovo modello salvato (val loss migliorata)")
        elif weighted_f1 >= best_f1_1:
            best_f1_1 = weighted_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_unet_for_f1.pth")
            print(" ==> Nuovo modello salvato (F1 migliorata)")
        else:
            patience_counter += 1
            print(f" ==> Nessun miglioramento. patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping attivato.")
                break
        
        print('')



# ---------------------------------------------------------------------------------------- #




'''
    Funzioni per estrarre coordinate discrete di keypoints da una heatmap.
    Utile per calcolare metriche come precision, recall, F1 a partire da PCK (Percentage of Correct Keypoints).    
'''


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
'''




def load_keypoints_from_csv(csv_path):
    """
    Carica i keypoints ground truth da un file CSV.

    Parameters
    ----------
    csv_path : str
        Percorso al file CSV contenente colonne 'x' e 'y'.

    Returns
    -------
    list of tuples
        Lista di (x, y) int.
    """
    df = pd.read_csv(csv_path)
    return list(zip(df['x'].astype(int), df['y'].astype(int)))




def extract_predicted_keypoints(heatmap, threshold=None, show_mask=False):
    """
    Estrae i keypoints predetti da una heatmap usando connected components + centroide.

    Parameters
    ----------
    heatmap : torch.Tensor
        Heatmap predetta di forma (1, H, W) o (H, W).
    threshold : float
        Valore di soglia per binarizzare la heatmap (default: 0.97).

    Returns
    -------
    list of lists
        Lista di [x, y] float: centri delle "macchie" sopra soglia.
    """
    
    # Preprocessing
    # Se è un tensore PyTorch, converti
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.squeeze().cpu().numpy()
    else:
        heatmap = np.squeeze(heatmap)
    
    if threshold is None:
        threshold = heatmap.max()*0.97
    else:
        threshold *= heatmap.max()
        
    binary = (heatmap > threshold).astype(np.uint8)
    
    if show_mask is True:
        plt.imshow(binary, cmap='gray')
        plt.title("Binarized Heatmap")
        plt.show()

    # Trova le componenti connesse (OpenCV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    keypoints = []
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        keypoints.append([int(round(cx)), int(round(cy))])

    return keypoints




def infer_keypoints_from_image(img_path, model, device='cuda', show_mask=False, threshold=None, npy=True, sigmoid=True):
    """
    Esegue inferenza su un'immagine e restituisce heatmap + keypoints predetti.

    Parameters
    ----------
    img_path : str
        Path all'immagine.
    model : torch.nn.Module
        Modello PyTorch già caricato e in `.eval()` mode.
    device : str
        'cuda' o 'cpu'.
    show_mask : bool
        Se True, mostra la binarizzazione della heatmap.
    threshold : float | None
        Threshold per la soglia (passato a extract_predicted_keypoints).

    Returns
    -------
    heatmap_pred : np.ndarray (H, W)
        Heatmap predetta.
    keypoints : list of [x, y]
        Coordinate dei keypoints in pixel.
    """
    # 1. Carica immagine grayscale e resize a 800x800
    if npy==False:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Immagine non trovata: {img_path}")
        img = cv2.resize(img, (800, 800))
        img = img.astype('float32') / 255.0
    else:
        img = np.load(img_path)  # shape atteso (H, W)
        img = img.astype(np.float32)  # assicurati che sia in float32
    
    # 2. Convertila in tensore torch [1, 1, H, W]
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Forward pass (inference)
    start_time = time.time()
    with torch.no_grad():
        output = model(img_tensor)  # output shape: [1, 1, 800, 800]
        if sigmoid == True: output = torch.sigmoid(output)  # Applica sigmoid per ottenere valori tra 0 e 1
        inference_time = time.time() - start_time
    heatmap_pred = output.squeeze().cpu().numpy()  # shape: (800, 800)

    # 4. Estrai keypoints
    keypoints = extract_predicted_keypoints(heatmap_pred, threshold=threshold, show_mask=show_mask)

    return heatmap_pred, keypoints, inference_time




def compute_pck_metrics(gt_points, pred_points, thresholds):
    """
    Calcola precision, recall e F1-score per varie soglie di distanza (PCK).

    Parametri:
    - gt_points (np.array di forma Nx2): Coordinate (x, y) dei keypoint ground truth.
    - pred_points (np.array di forma Mx2): Coordinate (x, y) dei keypoint predetti.
    - thresholds (iterabile o float/int): Soglie di distanza in pixel per il calcolo del PCK.

    Ritorna:
    - precisioni (list): Precisione per ciascuna soglia.
    - recall (list): Recall per ciascuna soglia.
    - f1_scores (list): F1-score per ciascuna soglia.
    """

    if len(gt_points) == 0 or len(pred_points) == 0:
        return [0.0] * (len(thresholds) if isinstance(thresholds, (list, np.ndarray)) else 1), \
               [0.0] * (len(thresholds) if isinstance(thresholds, (list, np.ndarray)) else 1), \
               [0.0] * (len(thresholds) if isinstance(thresholds, (list, np.ndarray)) else 1)

    if isinstance(thresholds, (list, np.ndarray)):
        precisions, recalls, f1_scores = [], [], []

        for t in thresholds:
            matched_gt = set()
            matched_pred = set()
            tp = 0

            for i, pred in enumerate(pred_points):
                pred = np.array(pred)
                for j, gt in enumerate(gt_points):
                    if j in matched_gt:
                        continue
                    gt = np.array(gt)
                    distance = np.linalg.norm(pred - gt)
                    if distance < t:
                        matched_gt.add(j)
                        matched_pred.add(i)
                        tp += 1
                        break

            fp = len(pred_points) - tp
            fn = len(gt_points) - tp

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)

        return precisions, recalls, f1_scores

    elif isinstance(thresholds, (int, float)):
        matched_gt = set()
        matched_pred = set()
        tp = 0

        for i, pred in enumerate(pred_points):
            pred = np.array(pred)
            for j, gt in enumerate(gt_points):
                if j in matched_gt:
                    continue
                gt = np.array(gt)
                distance = np.linalg.norm(pred - gt)
                if distance < thresholds:
                    matched_gt.add(j)
                    matched_pred.add(i)
                    tp += 1
                    break

        fp = len(pred_points) - tp
        fn = len(gt_points) - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        return [p], [r], [f1]  # restituisci comunque liste per compatibilità





def img_kp_pred_and_gr(keypoints_pred, keypoints_gt, img_path):
    
    # Carica immagine da file .npy (grayscale)
    img = np.load(img_path)
    img = img.astype(np.float32)
    
    # Normalizza se necessario
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # Converti in immagine colore per disegno
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Disegna ground truth in BLU
    for (x, y) in keypoints_gt:
        cv2.circle(img_color, (int(x), int(y)), 2, (255, 0, 0), -1)  # GT: BLU

    # Disegna predetti in VERDE
    for (x, y) in keypoints_pred:
        cv2.circle(img_color, (int(x), int(y)), 1, (0, 255, 0), -1)  # Pred: VERDE

    # Mostra immagine
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Predetti (verde) vs Ground Truth (blu)")
    plt.show()
    



def inference_dataset(
    datapath,
    output_path,
    model_path,
    device='cuda',
    pixel_thresholds=[2, 4, 6],
    attention_model=True,
    threshold=0.97,
    show_mask=False
):
    """
    Calcola le metriche PCK (precision, recall, F1) su un dataset di immagini + CSV GT.

    Parametri
    ----------
    datapath : str
        Cartella principale che contiene 'images/train' e 'centers/train'.
    model_path : str
        Percorso al file .pt/.pth del modello U-Net salvato.
    output_path : str
        Cartella dove salvare risultati (immagini annotate, heatmap, metriche).
    device : str
        'cuda' o 'cpu'.
    pixel_thresholds : list of int
        Soglie PCK (in pixel).
    attention_model : bool
        Se True, usa UNetWithAttention, altrimenti usa UNet base.
    threshold : float | None
        Soglia di binarizzazione per la heatmap (es. 0.3 = 30% max).
    show_mask : bool
        Se True, mostra le maschere binarie a video.

    Returns
    -------
    dict
        Contiene 'precision', 'recall', 'f1', 'stdmean_f1', 'inference_time' come liste o float.
    """

    os.makedirs(output_path, exist_ok=True)
    output_keypoints_dir = os.path.join(output_path, 'keypoints')
    output_heatmaps_dir = os.path.join(output_path, 'heatmaps')
    os.makedirs(output_keypoints_dir, exist_ok=True)
    os.makedirs(output_heatmaps_dir, exist_ok=True)

    model_cls = UNetWithAttention if attention_model else UNet
    model = model_cls(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img_dir = os.path.join(datapath, 'images', 'train')
    csv_dir = os.path.join(datapath, 'centers', 'train')

    image_paths = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.endswith('.png') or f.endswith('.npy')
    ])

    all_precisions = np.zeros(len(pixel_thresholds))
    all_recalls = np.zeros(len(pixel_thresholds))
    all_f1s = np.zeros(len(pixel_thresholds))
    all_f1_list = []  
    inference_time_total = 0.0
    count = 0

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        csv_path = os.path.join(csv_dir, base_name + '_centers.csv')
        if not os.path.exists(csv_path):
            continue

        gt_points = load_keypoints_from_csv(csv_path)
        heatmap, pred_points, inference_time = infer_keypoints_from_image(
            img_path, model, device=device, threshold=threshold, show_mask=show_mask
        )
        inference_time_total += inference_time
        precisions, recalls, f1s = compute_pck_metrics(
            np.array(gt_points), np.array(pred_points), pixel_thresholds
        )
        all_precisions += np.array(precisions)
        all_recalls += np.array(recalls)
        all_f1s += np.array(f1s)
        all_f1_list.append(f1s) 
        count += 1

        if img_path.endswith('.npy'):
            img = np.load(img_path).astype(np.float32)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)

        for (x, y) in gt_points:
            cv2.circle(img_color, (int(x), int(y)), 2, (255, 0, 0), -1)
        for (x, y) in pred_points:
            cv2.circle(img_color, (int(x), int(y)), 1, (0, 255, 0), -1)

        text = f"P: {precisions[1]:.2f}  R: {recalls[1]:.2f}  F1: {f1s[1]:.2f} | thr={pixel_thresholds[1]}px"
        cv2.putText(img_color, text, (10, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA)

        cv2.imwrite(os.path.join(output_keypoints_dir, base_name + '_pred.png'), img_color)
        heatmap_norm = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_heatmaps_dir, base_name + '_heatmap.png'), heatmap_norm)

    if count == 0:
        raise ValueError("Nessuna immagine valida trovata (nessun CSV corrispondente).")

    all_f1_array = np.array(all_f1_list)  # shape: (N immagini, len(pixel_thresholds))
    stdmean_f1 = (np.std(all_f1_array, axis=0) / np.sqrt(count)).tolist()

    return {
        'precision': (all_precisions / count).tolist(),
        'recall': (all_recalls / count).tolist(),
        'f1': (all_f1s / count).tolist(),
        'stdmean_f1': stdmean_f1,
        'inference_time': inference_time_total / count
    }