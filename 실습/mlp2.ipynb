import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

train_dataset = torchvision.datasets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True)

batch_size = 128

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.linear1 = nn.Linear(784, 784*3)
    self.linear2 = nn.Linear(784*3, 784*2)
    self.linear3 = nn.Linear(784*2, 10)

    self.activation = nn.Sigmoid()

  def forward(self, x):
    z1 = self.linear1(x)
    a1 = self.activation(z1)

    z2 = self.linear2(a1)
    a2 = self.activation(z2)

    z3 = self.linear3(a2)

    return z3

model = Model().to(device).train() 

optimizer = optim.SGD(model.parameters(), lr=0.1)

criterion = nn.CrossEntropyLoss()

epochs = 15

model.train() 

for epoch in range(epochs): 
  avg_cost = 0
  total_batch_num = len(train_dataloader)

  for b_x, b_y in train_dataloader:
    b_x = b_x.view(-1, 28*28).to(device) # 이미지크기 28*28 , 2차원 -> 1차원 평탄화
    logits = model(b_x)
    loss = criterion(logits, b_y.to(device))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_cost += loss / total_batch_num
  print('Epoch: {} / {}, cost : {}'.format(epoch+1, epochs, avg_cost))
