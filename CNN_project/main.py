import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

transform = transforms.Compose([transforms.Resize((100,100)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

train_dataset = torchvision.datasets.Flowers102(root="StanfordCars/",
                                                split='train',
                                                transform=transform,
                                                download=True)
test_dataset = torchvision.datasets.Flowers102(root="StanfordCars/",
                                               split='test',
                                               transform=transform,
                                               download=True)
sample = test_dataset[4][0].numpy()
sample = np.transpose(sample, (1,2,0))

fig, ax = plt.subplots(1,1)
ax.set_title('data')
ax.set_axis_off()
ax.imshow(sample)
# plt.show()

batch_size = 128
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

## AutoEncoder 모델코드 ##
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encode = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1), # 100 - 3 + 2 + 1 = 100
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, 2, 1), # (100 - 3 + 2)/2 + 1 = 50 
        nn.ReLU(),
    )
  def forward(self, input):
    return self.encode(input)

class Decoder(nn.Module):
  def zero_padding(self, input):
    batch_len, channel_len, width, height = input.size() # [128 32 50 50]
    len = width * 2 + 1

    unstrided_mat = torch.zeros(batch_len, channel_len, len, len)
    unstrided_mat[:,:,1::2,1::2] = input
    return unstrided_mat
  
  def __init__(self):
    super(Decoder, self).__init__()
    self.decode = nn.Sequential(
        nn.Conv2d(32, 16, 3, 1, 1), # 101 - 3 + 1 +2 = 101 
        nn.ReLU(),
        nn.Conv2d(16, 3, 2, 1), # 101 - 2 + 1 = 100
        nn.Tanh(),
    )

  def forward(self, input):
    input = self.zero_padding(input).to(device)
    return self.decode(input)

class AutoEncoder(nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
  def forward(self, input):
    z = self.encoder(input)
    x_hat = self.decoder(z)
    return z, x_hat

##### AutoEncoder 학습코드 ######

autoencoder = AutoEncoder().to(device).train()

criterion = nn.MSELoss()
cls_criterion = nn.CrossEntropyLoss()

optimizer_auto = optim.Adam(autoencoder.parameters(), lr=0.001) 

epochs = 30

print('----- autoencoder 학습 -----')
for epoch in range(epochs):
  autoencoder.train()
  avg_cost = 0
  total_batch_num = len(train_dataloader)

  for b_x, b_y in train_dataloader:
    b_x = b_x.to(device)
    z, b_x_hat = autoencoder(b_x)
    loss = criterion(b_x_hat, b_x)

    avg_cost += loss / total_batch_num
    optimizer_auto.zero_grad()
    loss.backward()
    optimizer_auto.step()
  print('Epoch : {} / {}, cost : {}'.format(epoch+1, epochs, avg_cost))

class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()
    self.classify = nn.Sequential(
        nn.Conv2d(32, 4000, kernel_size=50, stride=1),
        nn.ReLU(),
        nn.Conv2d(4000, 2000, kernel_size=1, stride=1),
    )
    self.linear = nn.Sequential(
        # 1. dropout 적용
        # nn.Dropout(0.7),
        # 3. batch norm 
        # nn.BatchNorm1d(2000),
        nn.Linear(2000, 102),
    )
  
  def forward(self, input):
    x = self.classify(input)
    x = torch.flatten(x, 1)
    x = self.linear(x)

    return x
  
classifier = Classifier().to(device).train()

optimizer = optim.Adam(
    [
        {"params": autoencoder.parameters(), "lr": 0.001},
        {"params": classifier.parameters(), "lr": 0.001},
    ]
)

autoencoder.eval()
classifier.train()
total_batch_num = len(train_dataloader)

epochs = 20

print('----- classifier 학습 -----')
for epoch in range(epochs):
  avg_cost = 0
  
  for b_x, b_y in train_dataloader:
    b_x = b_x.to(device)
    z1, b_x_hat = autoencoder(b_x)
    
    logits = classifier(z1)
    loss = cls_criterion(torch.flatten(logits, start_dim=1), b_y.to(device))

    # 2. L2 regularization 적용
    # reg = classifier.classify[0].weight.pow(2.0).sum()
    # reg += classifier.classify[2].weight.pow(2.0).sum()
    # loss += (0.003) * reg / len(b_x) / 2

    avg_cost += loss / total_batch_num

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print('Epoch : {} / {}, cost : {}'.format(epoch+1, epochs, avg_cost))


correct = 0
total = 0
classifier.eval()
autoencoder.eval()

for b_x, b_y in test_dataloader:
  b_x = b_x.to(device)
  with torch.no_grad():
    z1, b_x_hat = autoencoder(b_x)
    logits = classifier(z1)
  
  probs = nn.Softmax(dim = 1)(torch.flatten(logits, start_dim=1))
  predicts = torch.argmax(probs, dim = 1)
  
  total += len(b_y)
  correct += (predicts == b_y.to(device)).sum().item()
print(f'Accuracy of the network on test images: {100 * correct / total} %')
