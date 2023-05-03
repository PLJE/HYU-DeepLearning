import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

import torchvision
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.MNIST(root="MNIST_data/",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root="MNIST_data/",
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)

batch_size = 128

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1, 1,)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(32,64,3,1,1)
    self.fc1 = nn.Linear(64*7*7, 128)
    self.fc2 = nn.Linear(128, 10)
    self.activation = nn.ReLU()

  def forward(self, x):
    x = self.pool(self.activation(self.conv1(x)))
    x = self.pool(self.activation(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = self.activation(self.fc1(x))
    x = self.fc2(x)
    return x

model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

epochs = 30

model.train()

for epoch in range(epochs):
  model.train()
  avg_cost = 0
  total_batch_num = len(train_dataloader)

  for b_x, b_y in train_dataloader:
    logits = model(b_x.to(device))
    loss = criterion(logits, b_y.to(device))

    avg_cost += loss / total_batch_num
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print('Epoch : {} / {}, cost : {}'.format(epoch+1, epochs, avg_cost))

correct = 0
total = 0
model.eval()
for b_x, b_y in test_dataloader:
  with torch.no_grad():
    logits = model(b_x.to(device))
  probs = nn.Softmax(dim=1)(logits)
  predicts = torch.argmax(logits, dim=1)

  total += len(b_y)
  correct += (predicts == b_y.to(device)).sum().item()

print(f'Accuracy of the network on test images : {100 * correct // total} %')
