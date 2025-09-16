import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

## OUTPUT OF THE CODE
#100.0%
#100.0%
#100.0%
#100.0%
#Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
#Shape of y: torch.Size([64]) torch.int64
#Using mps device
#NeuralNetwork(
#  (flatten): Flatten(start_dim=1, end_dim=-1)
#  (linear_relu_stack): Sequential(
#    (0): Linear(in_features=784, out_features=512, bias=True)
#    (1): ReLU()
#    (2): Linear(in_features=512, out_features=512, bias=True)
#    (3): ReLU()
#    (4): Linear(in_features=512, out_features=10, bias=True)
#  )
#)
#Epoch 1
#-------------------------------
#loss: 2.304876  [   64/60000]
#loss: 2.289908  [ 6464/60000]
#loss: 2.267773  [12864/60000]
#loss: 2.265378  [19264/60000]
#loss: 2.255449  [25664/60000]
#loss: 2.214181  [32064/60000]
#loss: 2.238430  [38464/60000]
#loss: 2.193313  [44864/60000]
#loss: 2.188861  [51264/60000]
#loss: 2.161681  [57664/60000]
#Test Error: 
# Accuracy: 40.0%, Avg loss: 2.150826 
#
#Epoch 2
#-------------------------------
#loss: 2.162581  [   64/60000]
#loss: 2.150074  [ 6464/60000]
#loss: 2.087574  [12864/60000]
#loss: 2.106519  [19264/60000]
#loss: 2.061974  [25664/60000]
#loss: 1.990094  [32064/60000]
#loss: 2.041882  [38464/60000]
#loss: 1.946764  [44864/60000]
#loss: 1.951442  [51264/60000]
#loss: 1.888977  [57664/60000]
#Test Error: 
# Accuracy: 55.8%, Avg loss: 1.876820 
#
#Epoch 3
#-------------------------------
#loss: 1.914055  [   64/60000]
#loss: 1.873274  [ 6464/60000]
#loss: 1.756315  [12864/60000]
#loss: 1.802029  [19264/60000]
#loss: 1.690105  [25664/60000]
#loss: 1.639900  [32064/60000]
#loss: 1.685736  [38464/60000]
#loss: 1.570634  [44864/60000]
#loss: 1.595990  [51264/60000]
#loss: 1.505016  [57664/60000]
#Test Error: 
# Accuracy: 62.5%, Avg loss: 1.509818 
#
#Epoch 4
#-------------------------------
#loss: 1.579704  [   64/60000]
#loss: 1.532332  [ 6464/60000]
#loss: 1.388443  [12864/60000]
#loss: 1.461877  [19264/60000]
#loss: 1.345012  [25664/60000]
#loss: 1.337852  [32064/60000]
#loss: 1.366747  [38464/60000]
#loss: 1.280840  [44864/60000]
#loss: 1.312036  [51264/60000]
#loss: 1.226945  [57664/60000]
#Test Error: 
# Accuracy: 64.0%, Avg loss: 1.243428 
#
#Epoch 5
#-------------------------------
#loss: 1.321945  [   64/60000]
#loss: 1.292726  [ 6464/60000]
#loss: 1.134766  [12864/60000]
#loss: 1.236985  [19264/60000]
#loss: 1.121283  [25664/60000]
#loss: 1.138755  [32064/60000]
#loss: 1.170110  [38464/60000]
#loss: 1.099546  [44864/60000]
#loss: 1.133386  [51264/60000]
#loss: 1.065712  [57664/60000]
#Test Error: 
# Accuracy: 64.9%, Avg loss: 1.077634 
#
#Done!
#Saved PyTorch Model State to model.pth
#Predicted: "Ankle boot", Actual: "Ankle boot"