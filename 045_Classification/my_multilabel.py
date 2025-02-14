#%% imports
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim.adam
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import numpy as np
from collections import Counter

# %% data
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size=0.2)

# %% dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

ds_train = MultilabelDataset(X_train, y_train)
ds_test = MultilabelDataset(X_test, y_test)

dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=32, shuffle=True)

print(ds_train.X.shape)
print(ds_train.y.shape)

# %% model
class MultilabelModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
model = MultilabelModel(ds_train.X.shape[1], 10, ds_train.y.shape[1])

# %% training session
fn_loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
slope, bias = [], []
number_epochs = 100

for epoch in range(number_epochs):
    for (X,y) in dl_train:
        # optimization reset
        optimizer.zero_grad()
        # forward pass
        y_pred = model(X)
        # compute loss
        loss = fn_loss(y_pred, y)
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()
    if epoch % 10 == 0:
        print(f'epoch: {epoch}, loss: {loss.data.item()}')
        losses.append(loss.item())

# %% plot losses
sns.scatterplot(x=range(len(losses)), y=losses)

# %% test the model
with torch.no_grad():
    y_test_pred = model(X_test).round()

# %% naive classifier accuracy
y_test_str = [str(i) for i in y_test.detach().numpy()]
print(Counter(y_test_str).most_common())
most_common_cnt = Counter(y_test_str).most_common()[0][1]
print(f'Naive Classifier accuracy: {most_common_cnt/len(y_test_str)*100}%')

# %% test accuracy
print(f'Model accuracy: {accuracy_score(y_test, y_test_pred)*100}%')

