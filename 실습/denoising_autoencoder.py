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

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.linear = nn.Linear(784,256)
    self.activation = nn.Sigmoid()
  def forward(self, x):
    x = self.linear(x)
    x = self.activation(x)
    return x
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.linear = nn.Linear(256, 784)
    self.activation = nn.Sigmoid()
  def forward(self, x):
    x = self.linear(x)
    x = self.activation(x)
    return x
class AutoEncoder(nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
  def forward(self, x):
    z = self.encoder(x)
    x_hat = self.decoder(z)
    return z, x_hat

model = AutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
sample = test_dataset[1051][0].view(-1, 784).to(device)

epochs = 30

model.train()
for epoch in range(epochs):
  model.train()
  avg_cost = 0
  total_batch_num = len(train_dataloader)

  for b_x, b_y in train_dataloader:
    b_x = b_x.view(-1, 784).to(device)
    noise = torch.randn(b_x.shape).to(device) # noise
    noisy_b_x = b_x + noise

    z, b_x_hat = model(noisy_b_x)
    loss = criterion(b_x_hat, b_x)

    avg_cost += loss / total_batch_num
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print('Epoch : {} / {}, cost : {}'.format(epoch+1, epochs, avg_cost))

  #observe differences 
  model.eval()
  if epoch % 5 == 0:
    fig, ax = plt.subplots(1,3)
    with torch.no_grad():
      noise = torch.randn(sample.shape).to(device)
      noisy_sample = sample + noise
      test_z, test_output = model(noisy_sample)
    ax[0].set_title('x')
    ax[1].set_title('x_noise')
    ax[2].set_title('x_hat')

    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off(),
    ax[0].imshow(np.reshape(sample.detach().cpu(), (28,28)), cmap='gray')
    ax[1].imshow(np.reshape(noisy_sample.detach().cpu(), (28,28)), cmap='gray')
    ax[2].imshow(np.reshape(test_output.detach().cpu(), (28,28)), cmap='gray')
    plt.show()
