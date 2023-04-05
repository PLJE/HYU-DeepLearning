# https://wikidocs.net/58686
import torch
import torch.nn as nn
import torch.optim as optim

x_train = torch.FloatTensor([[0,0], [0,1], [1,0], [1,1]])
y_train = torch.FloatTensor([[0], [0], [0], [1]])

class LogisticRegression(nn.Module): # LogisticRegression 이 nn.Module 을 상속받음
  def __init__(self, x_in, x_out):
    super(LogisticRegression, self).__init__() # 부모 클래스의 init 호출
    self.linear = nn.Linear(x_in, x_out) # (입력의 차원, 출력의 차원)
    self.activation = nn.Sigmoid()
  def forward(self, x):
    z = self.linear(x)
    a = self.activation(z)
    return a

model = LogisticRegression(2, 1).train() # (입력의 차원, 출력의 차원)

optimizer = optim.SGD(model.parameters(), lr=0.01) # set optimizer

criterion = nn.BCELoss() # cross entropy loss

epochs = 8000
for epoch in range(epochs):
  model.train()
  hypothesis = model(x_train) # forward propagation
  cost = criterion(hypothesis+1e-8, y_train) # get cost
  optimizer.zero_grad()
  cost.backward() # backward propagation
  optimizer.step() # update parameters

  if epoch != 0 and epoch % 100 == 0:
    model.eval() # 평가 모드
    with torch.no_grad(): # gradient 계산 비활성화
      predicts = (model(x_train))
      print('predict with model : {}'.format(predicts))
      print('real value y : {}'.format(y_train))
