from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_datasets(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    test_set = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_set.classes
