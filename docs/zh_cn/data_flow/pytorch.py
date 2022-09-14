import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(nn.Flatten(),
                                               nn.Linear(28 * 28, 512),
                                               nn.ReLU(), nn.Linear(512, 512),
                                               nn.ReLU(), nn.Linear(512, 10))

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

training_data = datasets.MNIST(
    root='data', train=True, download=True, transform=ToTensor())
val_data = datasets.MNIST(
    root='data', train=False, download=True, transform=ToTensor())
train_dataloader = DataLoader(training_data, batch_size=64)
val_dataloader = DataLoader(val_data, batch_size=1)
train_length = len(train_dataloader.dataset)
val_length = len(val_dataloader.dataset)

model.train()
for iter, (inputs, labels) in enumerate(train_dataloader):
    inputs, labels = inputs.to(device), labels.to(device)
    pred = model(inputs)
    loss = loss_fn(pred, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        loss, current = loss.item(), iter * len(inputs)
        print(f'loss: {loss:>7f}  [{current:>5d}/{train_length:>5d}]')
# 验证
model.eval()
test_loss, correct = 0, 0
for inputs, labels in val_dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    pred = model(inputs)
    pred_label = pred.argmax(dim=1)
    correct += (pred_label == labels).type(torch.float).sum().item()
correct /= val_length
print(f'Test: \n Accuracy: {(100*correct):>0.1f}% \n')
