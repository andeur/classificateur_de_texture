
import torch
import torch.nn as nn
from torchvision import models

from donner import charger

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader, test_loader, classes = charger(batch_size=32)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
tete = list(model.fc.parameters())
corps = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
optimizer = torch.optim.Adam([
    {"params": corps, "lr": 1e-4},   # couches pré-entraînées : on y va doucement
    {"params": tete, "lr": 1e-3},    # nouvelle couche : on apprend plus vite
])
EPOQUES = 15
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOQUES)


def evaluer(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


meilleure_val = 0
for epoque in range(EPOQUES):
    model.train()
    perte_totale = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        perte_totale += loss.item() * y.size(0)
    scheduler.step()
    perte = perte_totale / len(train_loader.dataset)
    val = evaluer(val_loader)
    print(f"Époque {epoque + 1}: perte {perte:.3f}, précision val {val:.1%}")
    if val > meilleure_val:
        meilleure_val = val
        torch.save({"etat": model.state_dict(), "classes": classes},
                   "modele_finetune.pt")

# On recharge la meilleure version (choisie sur la validation) et on teste une seule fois
model.load_state_dict(torch.load("modele_finetune.pt")["etat"])
print(f"Meilleure val : {meilleure_val:.1%} | test : {evaluer(test_loader):.1%}")



