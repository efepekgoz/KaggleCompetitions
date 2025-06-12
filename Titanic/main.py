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

scaler = StandardScaler()
X = scaler.fit_transform(X)

y = data['Survived'].values.astype(np.float32)


X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

dataset= TensorDataset(X_tensor, y_tensor)

# 90-10-0 split: train and test only, no validation
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.1, random_state=42, stratify=y_tensor
)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# # Original 80-10-10 split with validation (commented out)
# X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# train_ds = TensorDataset(X_train, y_train)
# val_ds = TensorDataset(X_val, y_val)
# test_ds = TensorDataset(X_test, y_test)

# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=32)
# test_loader = DataLoader(test_ds, batch_size=32)

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
        
        if val_loader:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb)
                    loss = lossfn(preds, yb)
                    val_losses.append(loss.item())
            print(f"Epoch {e+1}, Val Loss: {np.mean(val_losses):.4f}")
        else:
            print(f"Epoch {e+1} completed.")

#train_model(model, train_loader, val_loader)
train_model(model, train_loader, None)

def eval_model(model, loader):
    model.eval()
    y_true=[]
    y_pred=[]

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).float()
            y_true.extend(yb.squeeze().numpy())
            y_pred.extend(predicted.squeeze().numpy())
    
    print("Accuracy: ",accuracy_score(y_true, y_pred))
    print("F1 Score: ",f1_score(y_true,y_pred))
    print("Classification Report:\n", classification_report(y_true,y_pred))

eval_model(model, test_loader)

### Submission ###

test_data = pd.read_csv('./Titanic/Data/test.csv')
passenger_ids = test_data['PassengerId']

test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Age'].fillna(data['Age'].mean(), inplace=True)   
test_data['Fare'].fillna(data['Fare'].mean(), inplace=True) 
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['Solo'] = (test_data['FamilySize'] == 1).astype(int)



X_kaggle_test = test_data[features].values
X_kaggle_test = scaler.transform(test_data[features].values)
X_kaggle_tensor = torch.tensor(X_kaggle_test, dtype=torch.float32)

model.eval()
with torch.no_grad():
    logits=model(X_kaggle_tensor)
    probs = torch.sigmoid(logits)
    preds = (probs>0.5).int().numpy().flatten()

submission=pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': preds

})
submission.to_csv('submission.csv',index=False)
print("submission.csv saved")
