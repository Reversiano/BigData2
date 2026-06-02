import torch

x = torch.tensor([1., 2., 3.])  #PyTorch è basato sui tensori, degli array n-dimensionali simili ai numpy array
print(x)  #stamperà un array di valori float

y = x.numpy()  #l'array pytorch viene convertito in array numpy
print(y)

x = x.type(torch.uint8)  #è possibile cambiare tipo agli array con questo comando
print(x)

device = "cuda" if torch.cuda.is_available() else "cpu"  #bisogna definire il device da usare per l'esecuzione dello script
print(device)


import nibabel as nib
from torch import nn
from tqdm import tqdm
import pdc 

volume = nib.load("941_S_6574_3D_MPRAGE.nii.gz")  #immagina medica digitale compressa (una generica è irreperibile a quanto pare)
neural_vol = volume.get_fdata()
print(neural_vol)  #stamperà l'array numpy risultante dalla conversione
neural_torch = torch.tensor(neural_vol)  #riconverte l'array in torch array
neural_torch = neural_torch.to(device)  #effettua l'esecuzione tramite il devide specificato


class Dataset(torch.utils.data.Dataset):  #dataset per caricare suddividere e organizzare i dati
    def __init__(self, paths, targets):  #inizializzatore/costruttore, salva i percorsi delle immagini e le etichette
        self.paths = paths
        self.targets = targets

    def __len__(self):  #misuratore 
        return len(self.paths)
    
    def __getitem__(self, index):  #estrattore, permette all'oggetto di essere indicizzato 
        scan_id = self.paths[index]
        image = pdc.read_image(scan_id)
        image = torch.tensor(image).float()
        y = torch.tensor(self.targets[index], dtype=torch.float)
        return image, y
    
training_data = Dataset(paths=x, targets=y)
#dataloader costruito per caricare correttamente i dati dopo la suddivisione
train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)  #shuffle casuale
#batch_size definisce il numero di campioni che verranno propagati attarevrso la rete
for batch_x, batch_y in tqdm(train_loader):
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)