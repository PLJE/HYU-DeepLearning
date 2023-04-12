import torchvision
import torchvision.transforms as transforms
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

transform = transforms.Compose(  #  평균값을 빼고, 표준편차로 나누어 줌으로써 각 채널의 픽셀값이 0을 중심으로 분포하도록 정규화
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4882, 0.4465), (0.247, 0.243, 0.261))]  # 첫번째인자 : CIFAR-10 데이터셋의 픽셀값의 평균(mean) 두번째인자 :  CIFAR-10 데이터셋의 픽셀값의 표준편차(standard deviation)
)

train_dataset = torchvision.datasets.CIFAR10(root="CIFAR10/",
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root="CIFAR10/",
                                             train=False,
                                             transform=transforms.ToTensor(),
                                             download=True)

batch_size = 128
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

class Model(nn.Module):
  def __init__(self, drop_prob):
    super(Model, self).__init__()
    self.linear1 = nn.Linear(32*32*3, 256)
    self.linear2 = nn.Linear(256, 128)
    self.linear3 = nn.Linear(128, 10)

    self.dropout = nn.Dropout(drop_prob)  # dropout regularization
    self.activation = nn.Sigmoid()

  def forward(self, x):
    z1 = self.linear1(x)
    a1 = self.activation(z1)
    a1 = self.dropout(a1)

    z2 = self.linear2(a1)
    a2 = self.activation(z2)
    a2 = self.dropout(a2)

    z3 = self.linear3(z2)

    return z3

model = Model(drop_prob=0.5).to(device).train()
optimizer = optim.SGD(model.parameters(), lr=1) 

criterion = nn.CrossEntropyLoss()

epochs = 70
lmbd = 0.003

train_avg_costs = []
test_avg_costs = []

test_total_batch = len(test_dataloader)
total_batch_num = len(train_dataloader)

for epoch in range(epochs):
  avg_cost = 0
  model.train()
  for b_x, b_y in train_dataloader:
    b_x = b_x.view(-1, 32*32*3).to(device)
    logits = model(b_x)
    loss = criterion(logits, b_y.to(device))

    reg = model.linear1.weight.pow(2.0).sum()  # L2 regularization
    reg += model.linear2.weight.pow(2.0).sum()
    reg += model.linear3.weight.pow(2.0).sum()
    
    loss += lmbd*reg/len(b_x)/2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_cost += loss / total_batch_num
  train_avg_costs.append(avg_cost.detach().cpu())
  print('Epoch : {} / {}, cost : {}'.format(epoch+1, epochs, avg_cost))

  test_avg_cost = 0
  model.eval()
  for b_x, b_y in test_dataloader:
    b_x = b_x.view(-1, 32*32*3).to(device)
    with torch.no_grad():
      logits = model(b_x)
      test_loss = criterion(logits, b_y.to(device))
    test_avg_cost += test_loss / test_total_batch
  test_avg_costs.append(test_avg_cost.detach().cpu())

epoch = range(epochs)
plt.plot(epoch, train_avg_costs, 'r-')
plt.plot(epoch, test_avg_costs, 'b-')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['train', 'test'])
plt.show()


