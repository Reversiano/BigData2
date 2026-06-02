import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchnn

# 1. Funzione per leggere i file originali del CIFAR-10
def unpickle(file):
    with open(file, 'rb') as fo:
        dizionario = pickle.load(fo, encoding='bytes')
    return dizionario

# 2. Il Dataset Personalizzato
class MioCIFAR10Completo(Dataset):
    def __init__(self, cartella_dataset, train=True, transform=None):
        self.transform = transform
        dati_grezzi = []
        self.etichette = []
        
        if train:
            nomi_file = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        else:
            nomi_file = ['test_batch']
            
        for nome_file in nomi_file:
            percorso_completo = os.path.join(cartella_dataset, nome_file)
            batch = unpickle(percorso_completo)
            dati_grezzi.append(batch[b'data'])
            self.etichette.extend(batch[b'labels'])
            
        dati_uniti = np.vstack(dati_grezzi)
        self.dati = dati_uniti.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

        self.dati_piatti = dati_uniti.reshape(-1, 3072).astype(np.float32) / 255.0  # Salviamo anche i dati "piatti" per usarli in Scikit-Learn (Gradient Boosting)

    def __len__(self):
        return len(self.dati)

    def __getitem__(self, idx):
        immagine = self.dati[idx]
        etichetta = self.etichette[idx]
        immagine_tensor = torch.from_numpy(immagine)
        
        if self.transform:
            immagine_tensor = self.transform(immagine_tensor)
            
        return immagine_tensor, etichetta

# 3. Il Wrapper per applicare la Data Augmentation solo al Training
class ApplicaTrasformazione(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        immagine, etichetta = self.subset[index]
        if self.transform:
            immagine = self.transform(immagine)
            
        return immagine, etichetta
        
    def __len__(self):
        return len(self.subset)

# 4. La Rete Neurale (La CNN)
class Cifar10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer convoluzionali
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Parte densa (Fully Connected) con Dropout 0.2 richiesto
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  #256 sorta di standard
        self.dropout = nn.Dropout(0.2) # Dropout aggiunto come richiesto
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Applichiamo il dropout
        x = self.fc2(x)
        return x
    
# --- CALLBACK CUSTOM PER IL CHECKPOINT ---
# Estendiamo l'Early Stopping del professore per salvare anche il modello
class CheckpointEarlyStopping(torchnn.EarlyStopping):
    def __init__(self, patience, min_delta, path_salvataggio):
        super().__init__(patience, min_delta)
        self.path_salvataggio = path_salvataggio
        
    # In PyTorch, passiamo i riferimenti di model e optimizer manualmente se non sono nel call
    def controlla_e_salva(self, validation_loss, model, optimizer, epoch):
        # Chiamiamo la logica originale del professore
        super().__call__(validation_loss)
        
        # Se c'è stato un miglioramento (la loss minima è stata appena aggiornata)
        if validation_loss == self.min_validation_loss:
            print(f"--> Miglioramento della loss ({validation_loss:.4f}). Salvataggio modello in corso...")
            torchnn.save_model(model, optimizer, epoch, [], [], [], [], {}, self.path_salvataggio)