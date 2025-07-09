import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

# Load data
train = pd.read_csv("./HousingPrices/Data/train.csv")
test = pd.read_csv("./HousingPrices/Data/test.csv")

test_ID = test["Id"]
y_train = np.log1p(train["SalePrice"])  # log-transform target
train.drop("SalePrice", axis=1, inplace=True)

# Combine train and test for preprocessing
all_data = pd.concat([train, test], keys=["train", "test"])

# Fill missing values
for col in all_data.columns:
    if all_data[col].dtype == "object":
        all_data[col] = all_data[col].fillna("None")
    else:
        all_data[col] = all_data[col].fillna(all_data[col].median())

# Encode categorical variables
for col in all_data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col])

# Split back
X_train = all_data.loc["train"]
X_test = all_data.loc["test"]

# Define base models
base_models = [
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("xgb", XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42)),
    ("lgb", LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42))
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Out-of-fold predictions for train and test
train_stack = np.zeros((X_train.shape[0], len(base_models)))
test_stack = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    print(f"Training base model: {name}")
    test_fold_preds = np.zeros((X_test.shape[0], 5))
    
    for j, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        train_stack[valid_idx, i] = val_pred
        
        test_fold_preds[:, j] = model.predict(X_test)
        
    test_stack[:, i] = test_fold_preds.mean(axis=1)

# Train meta-model
print("Training meta-model")
meta_model = Ridge(alpha=1.0)
meta_model.fit(train_stack, y_train)

meta_preds_log = meta_model.predict(test_stack)
meta_preds = np.expm1(meta_preds_log)

# Save submission
submission = pd.DataFrame({"Id": test_ID, "SalePrice": meta_preds})
submission.to_csv("./HousingPrices/submission.csv", index=False)

print("Stacking submission saved as submission.csv")
