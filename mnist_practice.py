import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Loading the data
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

# Defining a 2-layer fully conected model
class TwoLayerNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, hidden)   # layer 1
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 10)      # layer 2 (logits)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        return self.fc2(x)            # CrossEntropyLoss includes soft‑max


device = "cuda" if torch.cuda.is_available() else "cpu"
model  = TwoLayerNet().to(device)

loss_fn = nn.CrossEntropyLoss()
opt     = torch.optim.Adam(model.parameters(), lr=3e-4)

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    # quick validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_dl:
            preds = model(xb.to(device)).argmax(1)
            correct += (preds == yb.to(device)).sum().item()
            total   += yb.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1}: val accuracy {acc:.3f}")


torch.save(model.state_dict(), "mnist_two_layer.pth")

# quick confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt

model.eval(); preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        preds.extend(model(xb.to(device)).argmax(1).cpu())
        trues.extend(yb)

cm = confusion_matrix(trues, preds)
sns.heatmap(cm, annot=False, cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("MNIST 2‑layer Net")
plt.show()
