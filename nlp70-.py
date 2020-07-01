#72
y_train = np.loadtxt(base+'y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
loss = torch.nn.CrossEntropyLoss()
print (loss(torch.matmul(X_train[:1], W),y_train[:1]))
print (loss(torch.matmul(X_train[:4], W),y_train[:4]))

ans = [] # 以下、確認
for s,i in zip(softmax(torch.matmul(X_train[:4], W)),y_train[:4]):
  ans.append(-np.log(s[i]))
print (np.mean(ans))

クロスエントロピー損失の一部を実装し、確認をした。
#73. 確率的勾配降下法による学習

from torch.utils.data import TensorDataset, DataLoader
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 4),
        )
    def forward(self, X):
        return self.net(X)

model = LogisticRegression()
ds = TensorDataset(X_train, y_train)
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1)

for epoch in range(10):
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#74
def accuracy(pred, label):
  pred = np.argmax(pred.data.numpy(), axis=1)
  label = label.data.numpy()
  return (pred == label).mean()


X_valid = np.loadtxt(base+'X_valid.txt', delimiter=' ')
X_valid = torch.tensor(X_valid, dtype=torch.float32)
y_valid = np.loadtxt(base+'y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)

pred = model(X_train)
print (accuracy(pred, y_train))
pred = model(X_valid)
print (accuracy(pred, y_valid))
#75
%load_ext tensorboard
!rm -rf ./runs
%tensorboard --logdir ./runs
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from torch.utils.data import TensorDataset, DataLoader
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 4),
        )
    def forward(self, X):
        return self.net(X)

model = LogisticRegression()
ds = TensorDataset(X_train, y_train)
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1)

for epoch in range(10):
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
      y_pred = model(X_train)
      loss = loss_fn(y_pred, y_train)
      writer.add_scalar('Loss/train', loss, epoch)
      writer.add_scalar('Accuracy/train', accuracy(y_pred,y_train), epoch)

      y_pred = model(X_valid)
      loss = loss_fn(y_pred, y_valid)
      writer.add_scalar('Loss/valid', loss, epoch)
      writer.add_scalar('Accuracy/valid', accuracy(y_pred,y_valid), epoch)
#76
from torch.utils.data import TensorDataset, DataLoader
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 4),
        )
    def forward(self, X):
        return self.net(X)

model = LogisticRegression()
ds = TensorDataset(X_train, y_train)
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1)

for epoch in range(10):
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
      y_pred = model(X_train)
      loss = loss_fn(y_pred, y_train)
      writer.add_scalar('Loss/train', loss, epoch)
      writer.add_scalar('Accuracy/train', accuracy(y_pred,y_train), epoch)

      y_pred = model(X_valid)
      loss = loss_fn(y_pred, y_valid)
      writer.add_scalar('Loss/valid', loss, epoch)
      writer.add_scalar('Accuracy/valid', accuracy(y_pred,y_valid), epoch)

      torch.save(model.state_dict(), base+'output/'+str(epoch)+'.model')
      torch.save(optimizer.state_dict(), base+'output/'+str(epoch)+'.param')
#77
import time
from torch.utils.data import TensorDataset, DataLoader
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 4),
        )
    def forward(self, X):
        return self.net(X)

model = LogisticRegression()
ds = TensorDataset(X_train, y_train)
loss_fn = torch.nn.CrossEntropyLoss()


ls_bs = [2**i for i in range(15)]
ls_time = []
for bs in ls_bs:
  loader = DataLoader(ds, batch_size=bs, shuffle=True)
  optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1)
  for epoch in range(1):
      start = time.time()
      for xx, yy in loader:
          y_pred = model(xx)
          loss = loss_fn(y_pred, yy)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
  ls_time.append(time.time()-start)
print (ls_time)

#78
import time
from torch.utils.data import TensorDataset, DataLoader
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 4),
        )
    def forward(self, X):
        return self.net(X)

model = LogisticRegression()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

ds = TensorDataset(X_train.to(device), y_train.to(device))
loss_fn = torch.nn.CrossEntropyLoss()

ls_bs = [2**i for i in range(15)]
ls_time = []
for bs in ls_bs:
  loader = DataLoader(ds, batch_size=bs, shuffle=True)
  optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1)
  for epoch in range(1):
      start = time.time()
      for xx, yy in loader:
          y_pred = model(xx)
          loss = loss_fn(y_pred, yy)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
  ls_time.append(time.time()-start)
print (ls_time)

#79
import time
from torch.utils.data import TensorDataset, DataLoader
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 4),
        )
    def forward(self, X):
        return self.net(X)

model = MLP()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

ds = TensorDataset(X_train.to(device), y_train.to(device))
loss_fn = torch.nn.CrossEntropyLoss()

loader = DataLoader(ds, batch_size=1024, shuffle=True)
optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1)
for epoch in range(100):
    start = time.time()
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
      y_pred = model(X_train.to(device))
      loss = loss_fn(y_pred, y_train.to(device))
      writer.add_scalar('Loss/train', loss, epoch)
      train_acc = accuracy(y_pred.cpu(),y_train.cpu())
      writer.add_scalar('Accuracy/train', acc, epoch)

      y_pred = model(X_valid.to(device))
      loss = loss_fn(y_pred, y_valid.to(device))
      writer.add_scalar('Loss/valid', loss, epoch)
      valid_acc = accuracy(y_pred.cpu(),y_valid.cpu())
      writer.add_scalar('Accuracy/valid', acc, epoch)
print (train_acc, valid_acc)
