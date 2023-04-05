import torch
import torch.nn as nn
import torch.optim as optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

W = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

def criterion(y_hat, y):
  return torch.mean((y_hat - y) ** 2)

optimizer = optim.SGD([W,b], lr = 0.01)

epochs = 30
for epoch in range(epochs):
  hypothesis = x_train * W + b
  cost = criterion(hypothesis, y_train)

  optimizer.zero_grad() # Neural Network model 인스턴스를 만든 후, 역전파 단계를 실행하기 전에 변화도를 0으로 만든다.
  cost.backward() # backward propagation
  optimizer.step() # update parameters

  print('Epoch {:4d}/{} Cost: {:.6f} W: {:.3f}, b: {:.3f}'.format(epoch, epochs, cost.item(), W.item(), b.item()))
