
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MOYENNE = [0.485, 0.456, 0.406]
ECART = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MOYENNE, ECART),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MOYENNE, ECART),
])


def charger(batch_size=32):
    train = datasets.DTD("data", split="train", partition=1, download=True,
                         transform=transform_train)
    val = datasets.DTD("data", split="val", partition=1, download=True,
                       transform=transform_test)
    test = datasets.DTD("data", split="test", partition=1, download=True,
                        transform=transform_test)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
        train.classes,
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = charger()
    print(f"{len(train_loader.dataset)} images d'entraînement, "
          f"{len(val_loader.dataset)} de validation, "
          f"{len(test_loader.dataset)} de test")
    print(f"{len(classes)} classes :", classes[:10], "...")


