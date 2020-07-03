#80
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
train = pd.read_csv(base+'train.txt', header=None, sep='\t')
valid = pd.read_csv(base+'valid.txt', header=None, sep='\t')
test = pd.read_csv(base+'test.txt', header=None, sep='\t')
vectorizer = CountVectorizer(min_df=2)
train_title = train.iloc[:,1].str.lower()
cnt = vectorizer.fit_transform(train_title).toarray()
sm = cnt.sum(axis=0)
idx = np.argsort(sm)[::-1]
words = np.array(vectorizer.get_feature_names())[idx]
d = dict()
for i in range(len(words)):
  d[words[i]] = i+1
def get_id(sentence):
  r = []
  for word in sentence:
    r.append(d.get(word,0))
  return r

def df2id(df):
  ids = []
  for i in df.iloc[:,1].str.lower():
    ids.append(get_id(i.split()))
  return ids

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

#81
import torch
dw = 300
dh = 50
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(len(words)+1,dw)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:,-1,:] # 最後のステップ
        y = self.linear(y)
        y = self.softmax(y)
        return y

#82
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
%load_ext tensorboard
!rm -rf ./runs
%tensorboard --logdir ./runs
writer = SummaryWriter()

max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:,-1,:] # 最後のステップ
        y = self.linear(y)
        # y = self.softmax(y) # torch.nn.CrossEntropyLoss()がsoftmaxは含む
        return y

def list2tensor(data, max_len):
  new = []
  for d in data:
    if len(d) > max_len:
      d = d[:max_len]
    else:
      d += [PAD] * (max_len - len(d))
    new.append(d)
  return torch.tensor(new, dtype=torch.int64)

def accuracy(pred, label):
  pred = np.argmax(pred.data.numpy(), axis=1)
  label = label.data.numpy()
  return (pred == label).mean()



train = pd.read_csv(base+'train.txt', header=None, sep='\t')
valid = pd.read_csv(base+'valid.txt', header=None, sep='\t')
test = pd.read_csv(base+'test.txt', header=None, sep='\t')

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = np.loadtxt(base+'y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt(base+'y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt(base+'y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)


model = RNN()
ds = TensorDataset(X_train, y_train)
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(2):
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
      print (accuracy(y_pred,y_train))

      y_pred = model(X_valid)
      loss = loss_fn(y_pred, y_valid)
      writer.add_scalar('Loss/valid', loss, epoch)
      writer.add_scalar('Accuracy/valid', accuracy(y_pred,y_valid), epoch)
      print (accuracy(y_pred,y_valid))
#83
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
%load_ext tensorboard
!rm -rf ./runs
%tensorboard --logdir ./runs
writer = SummaryWriter()

max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:,-1,:] # 最後のステップ
        y = self.linear(y)
        # y = self.softmax(y) # torch.nn.CrossEntropyLoss()がsoftmaxは含む
        return y

def list2tensor(data, max_len):
  new = []
  for d in data:
    if len(d) > max_len:
      d = d[:max_len]
    else:
      d += [PAD] * (max_len - len(d))
    new.append(d)
  return torch.tensor(new, dtype=torch.int64)

def accuracy(pred, label):
  pred = np.argmax(pred.data.to('cpu').numpy(), axis=1)
  label = label.data.to('cpu').numpy()
  return (pred == label).mean()

train = pd.read_csv(base+'train.txt', header=None, sep='\t')
valid = pd.read_csv(base+'valid.txt', header=None, sep='\t')
test = pd.read_csv(base+'test.txt', header=None, sep='\t')

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = np.loadtxt(base+'y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt(base+'y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt(base+'y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)


model = RNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
ds = TensorDataset(X_train.to(device), y_train.to(device))
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1024, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(10):
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
      writer.add_scalar('Accuracy/train', accuracy(y_pred,y_train), epoch)
      print (accuracy(y_pred,y_train))

      y_pred = model(X_valid.to(device))
      loss = loss_fn(y_pred, y_valid.to(device))
      writer.add_scalar('Loss/valid', loss, epoch)
      writer.add_scalar('Accuracy/valid', accuracy(y_pred,y_valid), epoch)
      print (accuracy(y_pred,y_valid))

#84
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:,-1,:] # 最後のステップ
        y = self.linear(y)
        # y = self.softmax(y) # torch.nn.CrossEntropyLoss()がsoftmaxは含む
        return y

def list2tensor(data, max_len):
  new = []
  for d in data:
    if len(d) > max_len:
      d = d[:max_len]
    else:
      d += [PAD] * (max_len - len(d))
    new.append(d)
  return torch.tensor(new, dtype=torch.int64)

def accuracy(pred, label):
  pred = np.argmax(pred.data.to('cpu').numpy(), axis=1)
  label = label.data.to('cpu').numpy()
  return (pred == label).mean()


train = pd.read_csv(base+'train.txt', header=None, sep='\t')
valid = pd.read_csv(base+'valid.txt', header=None, sep='\t')
test = pd.read_csv(base+'test.txt', header=None, sep='\t')

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = np.loadtxt(base+'y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt(base+'y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt(base+'y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)

model = RNN()
#print (model.emb.weight)
for k,v in d.items():
  if k in w2v.vocab:
    #v = np.random.randint(1,PAD) #ランダムな単語ベクトルでも効果があるか
    model.emb.weight[v] = torch.tensor(w2v[k], dtype=torch.float32)
#print (model.emb.weight)
model.emb.weight = torch.nn.Parameter(model.emb.weight)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

ds = TensorDataset(X_train.to(device), y_train.to(device))
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1024, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(10):
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
      writer.add_scalar('Accuracy/train', accuracy(y_pred,y_train), epoch)
      print (accuracy(y_pred,y_train))

      y_pred = model(X_valid.to(device))
      loss = loss_fn(y_pred, y_valid.to(device))
      writer.add_scalar('Loss/valid', loss, epoch)
      writer.add_scalar('Accuracy/valid', accuracy(y_pred,y_valid), epoch)
      print (accuracy(y_pred,y_valid))
#print (model.emb.weight)
#85
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.rnn1 = torch.nn.RNN(dw,dh,bidirectional=True,batch_first=True)
        self.rnn2 = torch.nn.RNN(2*dh,dh,bidirectional=True,batch_first=True)
        self.rnn3 = torch.nn.RNN(2*dh,dh,bidirectional=True,batch_first=True)
        self.linear = torch.nn.Linear(2*dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn1(x, h)
        y, h = self.rnn2(y, h)
        y, h = self.rnn3(y, h)
        y = y[:,-1,:] # 最後のステップ
        y = self.linear(y)
        # y = self.softmax(y) # torch.nn.CrossEntropyLoss()がsoftmaxは含む
        return y

def list2tensor(data, max_len):
  new = []
  for d in data:
    if len(d) > max_len:
      d = d[:max_len]
    else:
      d += [PAD] * (max_len - len(d))
    new.append(d)
  return torch.tensor(new, dtype=torch.int64)

def accuracy(pred, label):
  pred = np.argmax(pred.data.to('cpu').numpy(), axis=1)
  label = label.data.to('cpu').numpy()
  return (pred == label).mean()


train = pd.read_csv(base+'train.txt', header=None, sep='\t')
valid = pd.read_csv(base+'valid.txt', header=None, sep='\t')
test = pd.read_csv(base+'test.txt', header=None, sep='\t')

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = np.loadtxt(base+'y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt(base+'y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt(base+'y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)

model = RNN()
#print (model.emb.weight)
for k,v in d.items():
  if k in w2v.vocab:
    #v = np.random.randint(1,PAD) #ランダムな単語ベクトルでも効果があるか
    model.emb.weight[v] = torch.tensor(w2v[k], dtype=torch.float32)
#print (model.emb.weight)
model.emb.weight = torch.nn.Parameter(model.emb.weight)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

ds = TensorDataset(X_train.to(device), y_train.to(device))
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1024, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(10):
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
      writer.add_scalar('Accuracy/train', accuracy(y_pred,y_train), epoch)
      print (accuracy(y_pred,y_train))

      y_pred = model(X_valid.to(device))
      loss = loss_fn(y_pred, y_valid.to(device))
      writer.add_scalar('Loss/valid', loss, epoch)
      writer.add_scalar('Accuracy/valid', accuracy(y_pred,y_valid), epoch)
      print (accuracy(y_pred,y_valid))
      
#86
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.conv = torch.nn.Conv1d(dw,dh,3,padding=1) # in_channels:dw, out_channels: dh
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(max_len)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.pool(x)
        x = x.view(x.shape[0], x.shape[1])
        y = self.linear(x)
        y = self.softmax(y) # torch.nn.CrossEntropyLoss()がsoftmaxは含む
        return y

def list2tensor(data, max_len):
  new = []
  for d in data:
    if len(d) > max_len:
      d = d[:max_len]
    else:
      d += [PAD] * (max_len - len(d))
    new.append(d)
  return torch.tensor(new, dtype=torch.int64)

def accuracy(pred, label):
  pred = np.argmax(pred.data.to('cpu').numpy(), axis=1)
  label = label.data.to('cpu').numpy()
  return (pred == label).mean()


train = pd.read_csv(base+'train.txt', header=None, sep='\t') 
valid = pd.read_csv(base+'valid.txt', header=None, sep='\t') 
test = pd.read_csv(base+'test.txt', header=None, sep='\t') 

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = np.loadtxt(base+'y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt(base+'y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt(base+'y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)

model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
with torch.no_grad():
  y_pred = model(X_train.to(device))

#87
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.conv = torch.nn.Conv1d(dw,dh,3,padding=1) # in_channels:dw, out_channels: dh
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(max_len)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.pool(x)
        x = x.view(x.shape[0], x.shape[1])
        y = self.linear(x)
        return y

def list2tensor(data, max_len):
  new = []
  for d in data:
    if len(d) > max_len:
      d = d[:max_len]
    else:
      d += [PAD] * (max_len - len(d))
    new.append(d)
  return torch.tensor(new, dtype=torch.int64)

def accuracy(pred, label):
  pred = np.argmax(pred.data.to('cpu').numpy(), axis=1)
  label = label.data.to('cpu').numpy()
  return (pred == label).mean()


train = pd.read_csv(base+'train.txt', header=None, sep='\t') 
valid = pd.read_csv(base+'valid.txt', header=None, sep='\t') 
test = pd.read_csv(base+'test.txt', header=None, sep='\t') 

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = np.loadtxt(base+'y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt(base+'y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt(base+'y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)

model = CNN()
#print (model.emb.weight)
for k,v in d.items():
  if k in w2v.vocab:
    #v = np.random.randint(1,PAD) #ランダムな単語ベクトルでも効果があるか
    model.emb.weight[v] = torch.tensor(w2v[k], dtype=torch.float32)
#print (model.emb.weight)

model.emb.weight = torch.nn.Parameter(model.emb.weight)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

ds = TensorDataset(X_train.to(device), y_train.to(device))
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1024, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
for epoch in range(10):
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
      writer.add_scalar('Accuracy/train', accuracy(y_pred,y_train), epoch)
      print (accuracy(y_pred,y_train))

      y_pred = model(X_valid.to(device))
      loss = loss_fn(y_pred, y_valid.to(device))
      writer.add_scalar('Loss/valid', loss, epoch)
      writer.add_scalar('Accuracy/valid', accuracy(y_pred,y_valid), epoch)
      print (accuracy(y_pred,y_valid))

#88	
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.conv = torch.nn.Conv1d(dw,dh,3,padding=1) # in_channels:dw, out_channels: dh
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(max_len)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.pool(x)
        x = x.view(x.shape[0], x.shape[1])
        y = self.linear(x)
        return y

def list2tensor(data, max_len):
  new = []
  for d in data:
    if len(d) > max_len:
      d = d[:max_len]
    else:
      d += [PAD] * (max_len - len(d))
    new.append(d)
  return torch.tensor(new, dtype=torch.int64)

def accuracy(pred, label):
  pred = np.argmax(pred.data.to('cpu').numpy(), axis=1)
  label = label.data.to('cpu').numpy()
  return (pred == label).mean()


train = pd.read_csv(base+'train.txt', header=None, sep='\t') 
valid = pd.read_csv(base+'valid.txt', header=None, sep='\t') 
test = pd.read_csv(base+'test.txt', header=None, sep='\t') 

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = np.loadtxt(base+'y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt(base+'y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt(base+'y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)

model = CNN()
#print (model.emb.weight)
for k,v in d.items():
  if k in w2v.vocab:
    #v = np.random.randint(1,PAD) #ランダムな単語ベクトルでも効果があるか
    model.emb.weight[v] = torch.tensor(w2v[k], dtype=torch.float32)
#print (model.emb.weight)

model.emb.weight = torch.nn.Parameter(model.emb.weight)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

ds = TensorDataset(X_train.to(device), y_train.to(device))
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1024, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(50):
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
      writer.add_scalar('Accuracy/train', accuracy(y_pred,y_train), epoch)
      print (accuracy(y_pred,y_train))

      y_pred = model(X_valid.to(device))
      loss = loss_fn(y_pred, y_valid.to(device))
      writer.add_scalar('Loss/valid', loss, epoch)
      writer.add_scalar('Accuracy/valid', accuracy(y_pred,y_valid), epoch)
      print (accuracy(y_pred,y_valid))

#89
!pip install transformers
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import *
import torch.nn as nn
import torch.nn.functional as F

max_len = 15
PAD = 0
n_unit =  768

tokenizer_class = BertTokenizer
tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


class BertClassifier(nn.Module):
    def __init__(self, n_classes=4):
        super(BertClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased') 
        self.fc = nn.Linear(n_unit, n_classes)

    def forward(self, ids):
        seg_ids = torch.zeros_like(ids) # 全て同一セグメントとみなす
        attention_mask = (ids > 0)
        last_hidden_state, _ = self.bert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
        x = last_hidden_state[:,0,:] # CLSトークン
        logit = self.fc(x.view(-1,n_unit))
        return logit


def list2tensor(data, max_len):
  new = []
  for d in data:
    if len(d) > max_len:
      d = d[:max_len]
    else:
      d += [PAD] * (max_len - len(d))
    new.append(d)
  return torch.tensor(new, dtype=torch.int64)

def accuracy(pred, label):
  pred = np.argmax(pred.data.to('cpu').numpy(), axis=1)
  label = label.data.to('cpu').numpy()
  return (pred == label).mean()

def df2id(df):
  tokenized = df[1].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
  return tokenized

train = pd.read_csv(base+'train.txt', header=None, sep='\t') 
valid = pd.read_csv(base+'valid.txt', header=None, sep='\t') 
test = pd.read_csv(base+'test.txt', header=None, sep='\t') 

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = np.loadtxt(base+'y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt(base+'y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt(base+'y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)
model = BertClassifier()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

dfs_freeze(model)
model.fc.requires_grad_(True)

ds = TensorDataset(X_train.to(device), y_train.to(device))
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1024, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(30):
    print(epoch)
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
      writer.add_scalar('Accuracy/train', accuracy(y_pred,y_train), epoch)
      print (accuracy(y_pred,y_train))

      y_pred = model(X_valid.to(device))
      loss = loss_fn(y_pred, y_valid.to(device))
      writer.add_scalar('Loss/valid', loss, epoch)
      writer.add_scalar('Accuracy/valid', accuracy(y_pred,y_valid), epoch)
      print (accuracy(y_pred,y_valid))
