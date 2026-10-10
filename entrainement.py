
import torch
import torch.nn as nn
from torchvision import models

from donner import charger

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Appareil :", device)

train_loader, val_loader, test_loader, classes = charger(batch_size=32)

# ResNet18 pré-entraîné : on gèle tout, puis on remplace la dernière couche
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)


def evaluer(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


for epoque in range(10):
    model.train()
    perte_totale = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        perte_totale += loss.item() * y.size(0)
    perte = perte_totale / len(train_loader.dataset)
    print(f"Époque {epoque + 1}: perte {perte:.3f}, précision val {evaluer(val_loader):.1%}")

print(f"Précision sur le test : {evaluer(test_loader):.1%}")
torch.save({"etat": model.state_dict(), "classes": classes}, "modele_textures.pt")




