import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import random_split

# Importiamo i nostri due file!
import modelli_e_dati
import torchnn  # Assicurati di aver rinominato il file del prof togliendo il trattino -


#PUNTO 1
def main():
    # 1. IMPOSTAZIONI INIZIALI
    cartella_dati = './cifar-10-batches-py' # Controlla che questo percorso sia corretto!
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Stiamo usando il dispositivo: {dispositivo}")

    # 2. FASE DATI (Usiamo le classi del nostro file modelli_e_dati.py)
    # Definiamo la Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),      # Mirroring orizzontale casuale
        transforms.RandomVerticalFlip(),        # Mirroring verticale casuale
        transforms.RandomAffine(                # Random....Flip e RandomAffine lavorano su immagini in formato PIL (Python Imaging Library)
            degrees=[-10, 10],                  # Rotazione casuale nel range [-10°, +10°]
            translate=(5/32, 5/32)              # Traslazione casuale in x e y di +-5 pixel
        )
    ])

    # Carichiamo il Training Set da 50.000 immagini
    train_set = modelli_e_dati.MioCIFAR10Completo(cartella_dati, train=True, transform=None)

    # Carichiamo il Test Set
    test_set_init = modelli_e_dati.MioCIFAR10Completo(cartella_dati, train=False, transform=None)
    
    # Lo spacchiamo in Training Set puro (90% = 45k) e Validation Set (10% = 5k)
    train_subset, val_subset = random_split(train_set, [45000, 5000])
    
    # Applichiamo l'augmentation SOLO al train puro
    train_set_augmented = modelli_e_dati.ApplicaTrasformazione(train_subset, transform=train_transform)
    # Mentre validation set e test set rimangono invariati
    val_set = modelli_e_dati.ApplicaTrasformazione(val_subset, transform=None)
    test_set = modelli_e_dati.ApplicaTrasformazione(test_set_init, transform=None)

######################################################################################################################################################################################################
#PUNTO 2: GRADIENT BOOSTING (Scikit-Learn)

    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score

    print("--- Avvio Gradient Boosting ---")
    # Estraiamo i dati piatti dal dataset originale (senza augmentation per il ML classico)
    X_train_ml = train_set.dati_piatti
    y_train_ml = np.array(train_set.etichette)
    X_test_ml = test_set.subset.dati_piatti
    y_test_ml = np.array(test_set.subset.etichette)

    # Parametri richiesti dalla traccia
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_iter': [50, 100, 200], # max_iter equivale a n_estimators nell'HistGradientBoosting
        'min_samples_leaf': [10, 20, 50]
    }
    
    clf = HistGradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(
        clf, 
        param_grid, 
        cv=3,       # <-- CAMBIA QUESTO DA 10 A 3 
        n_jobs=-1, 
        verbose=3
    )
    
    # NOTA: Questa istruzione richiederà MOLTO tempo. 
    # Per provare il codice rapidamente, scommenta la riga sotto e commenta il fit completo.
    grid_search.fit(X_train_ml[:1000], y_train_ml[:1000]) 
    #grid_search.fit(X_train_ml, y_train_ml)
    
    best_gb = grid_search.best_estimator_
    print(f"Migliori Parametri GB: {grid_search.best_params_}")
    
    # Accuratezza del miglior modello
    acc_train_clf = accuracy_score(y_train_ml, best_gb.predict(X_train_ml))
    acc_test_clf = accuracy_score(y_test_ml, best_gb.predict(X_test_ml))
    print(f"GB - Accuracy Addestramento: {acc_train_clf:.4f}")
    print(f"GB - Accuracy Predizione (Test): {acc_test_clf:.4f}")

######################################################################################################################################################################################################
#PUNTO 3: Rete Neurale

    # 3. CREIAMO I DATALOADER (Usando la funzione del professore!)
    print("Creazione dei DataLoader in corso...")
    trainloader, valloader, testloader = torchnn.make_dataloaders(
        train_data=train_set_augmented,
        val_data=val_set,
        test_data=test_set,
        batch=64 # Passiamo 64 immagini alla volta
    )

    # 4. CREIAMO IL MODELLO (Il "Dipendente")
    modello = modelli_e_dati.Cifar10CNN()
    modello.to(dispositivo)

    # 5. CREIAMO LOSS E OPTIMIZER (Il "Supervisore")
    loss_fn = nn.CrossEntropyLoss()
    #ottimizzatore = optim.Adam(modello.parameters(), lr=0.001)
    opt = optim.SGD(modello.parameters(), lr=0.01, nesterov=True, momentum=0.9)  #momentum a 0.9 per equaz. di Nesterov, per True deve essere =! 0

    # SCHEDULER per far decrescere esponenzialmente il learing rate
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)  #gamma pari al momentum

    # EARLY STOP CHECKPOINT (ref a modelli_e_dati.py)
    early_stop_checkpoint = modelli_e_dati.CheckpointEarlyStopping(patience=5, min_delta=0.01, path_salvataggio="best-model.pth")

    # 6. ADDESTRAMENTO (Usando la mega-funzione del professore!)
    print("Inizio Addestramento...")
    train_loss, val_loss, test_loss, accuracy, test_metrics = torchnn.train_test(
        model=modello,
        optimizer=opt,
        device=dispositivo,
        train_dataloader=trainloader,
        val_dataloader=valloader,
        test_dataloader=testloader,
        epochs=30,  #limite massimo per cui il programma può girare, dovuto all'early stopping
        train_loss_fn=loss_fn,
        test_loss_fn=loss_fn
    )

    # 7. VISUALIZZAZIONE RISULTATI (Usando le funzioni del professore)
    torchnn.displayLosses(train_loss, test_loss, val_loss)
    torchnn.displayMetrics(accuracy, test_metrics)

######################################################################################################################################################################################################
#PUNTO 4: Lista metriche e grafici

    from sklearn.metrics import confusion_matrix, roc_auc_score
    from  sklearn.preprocessing import label_binarize

    # 4.1 Conservare la lista accuracy e stamparla (già fatto dalla train_test del prof)
    torchnn.displayMetrics(accuracy, test_metrics)
    
    # Ottenere predizioni CNN per la matrice di confusione e ROC
    modello.eval()
    y_true_nn = []
    y_pred_nn = []
    y_prob_nn = []
    
    with torch.no_grad():
        for X, y in testloader:
            X = X.to(dispositivo)
            logits = modello(X)
            probs = torch.softmax(logits, dim=1)
            
            y_true_nn.extend(y.cpu().numpy())
            y_pred_nn.extend(logits.argmax(1).cpu().numpy())
            y_prob_nn.extend(probs.cpu().numpy())
            
    y_true_nn = np.array(y_true_nn)
    y_prob_nn = np.array(y_prob_nn)
    
    # Matrici di Confusione
    print("Matrice Confusione GB:\n", confusion_matrix(y_test_ml, best_gb.predict(X_test_ml)))
    print("Matrice Confusione CNN:\n", confusion_matrix(y_true_nn, y_pred_nn))
    
    # Binarizza le labels per calcolare ROC/AUC Multiclasse (One-vs-Rest)
    y_test_bin = label_binarize(y_test_ml, classes=range(10))
    y_prob_gb = best_gb.predict_proba(X_test_ml)
    
    # Calcolo e stampa AUC per singola classe
    for i in range(10):
        auc_gb = roc_auc_score(y_test_bin[:, i], y_prob_gb[:, i])
        auc_nn = roc_auc_score(y_test_bin[:, i], y_prob_nn[:, i])
        print(f"Classe {i} - AUC GB: {auc_gb:.4f} | AUC CNN: {auc_nn:.4f}")

######################################################################################################################################################################################################

if __name__ == '__main__':
    # Questo comando serve su Windows per far funzionare bene i DataLoader
    main()