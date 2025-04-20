import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH = 128

train_ds = datasets.MNIST(
    root="data", train=True,  download=True,
    transform=transforms.ToTensor()
)
test_ds  = datasets.MNIST(
    root="data", train=False, download=True,
    transform=transforms.ToTensor()
)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH)
