import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./Titanic/Data/train.csv")
data = data[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']]
data['Sex'] = data['Sex'].map({'male':0,'female':1})
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
data['FamilySize'] = data['SibSp']+data['Parch']+1
data['Solo']=(data['FamilySize']==1).astype(int)

features=['Pclass','Sex','Age','Fare','FamilySize','Solo']

X = data[features].values
y = data['Survived'].values.astype(np.float32)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

dataset= TensorDataset(X_tensor, y_tensor)
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(64,32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(32,1),
        )
    def forward(self,x):
        return self.net(x)

model = FCNN(input_dim=X.shape[1])
lossfn=nn.BCEWithLogitsLoss()
#optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
optimiser= torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

def train_model(model, train_loader, val_loader, epochs=20):
    for e in range(epochs):
        model.train()
        for xb, yb in train_loader:
            preds = model(xb)
            loss=lossfn(preds,yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb,yb in val_loader:
                preds=model(xb)
                loss=lossfn(preds, yb)
                val_losses.append(loss.item())
        print(f"Epoch {e+1}, Val Loss: {np.mean(val_losses):.4f}")

train_model(model, train_loader, val_loader)

def eval_model(model, loader):
    model.eval()
    y_true=[]
    y_pred=[]

    with torch.no_grad():
        for xb,yb in loader:
            preds= model(xb)
            predicted = (preds>0.5).float()
            y_true.extend(yb.squeeze().numpy())
            y_pred.extend(predicted.squeeze().numpy())
    
    print("Accuracy: ",accuracy_score(y_true, y_pred))
    print("F1 Score: ",f1_score(y_true,y_pred))
    print("Classification Report:\n", classification_report(y_true,y_pred))

eval_model(model, test_loader)
