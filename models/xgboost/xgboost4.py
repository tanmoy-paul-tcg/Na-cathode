import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import torch
import xgboost
from sklearn.impute import SimpleImputer

df = pd.read_csv('trainingData.csv')

exclude_columns = ['Discharge_ID','Formula_discharge']  #

# searate target variable
X = df.drop(columns=['Average_Voltage_(V)'] + exclude_columns)
y = df['Average_Voltage_(V)']

imputer = SimpleImputer(strategy='mean') # median, most_frequent
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

y_numpy = y_tensor.numpy()

dtrain = xgboost.DMatrix(X_tensor, label=y_numpy)

# parameters for GridSearch
param_grid = {
    'n_estimators': [100, 150, 200, 300, 400, 500, 550, 600, 800, 1000, 1200],
    'learning_rate': [0.002, 0.01, 0.05, 0.9, 0.5, 0.3],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 9, 12], 
    'min_child_weight': [1, 3, 5, 6, 8, 9, 20, 50],  
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],  
    'reg_lambda':[0.1, 1, 5, 50],
    'reg_alpha':[0.1, 0.6, 1, 10, 50],
    'subsample': [0.5, 0.6, 0.8, 1.0],  # >= 0.5 for good results.

}

xgb_regressor = XGBRegressor(objective='reg:squarederror', random_state=20)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, cv=8, scoring='neg_mean_absolute_error')
grid_search.fit(X_scaled, y)
best_params = grid_search.best_params_

# print("Best Hyperparameters:", best_params) #

# XGBoost regressor with the best hyperparameters
best_xgb_regressor = XGBRegressor(n_estimators=best_params['n_estimators'],
                                  learning_rate=best_params['learning_rate'],
                                  max_depth=best_params['max_depth'],
                                  min_child_weight=best_params['min_child_weight'],
                                  gamma=best_params['gamma'],
                                  reg_lambda=best_params['reg_lambda'],
                                  reg_alpha=best_params['reg_alpha'],
                                  subsample=best_params['subsample'],
                                  objective='reg:squarederror',
                                  random_state=50)

# 8-fold cross-validation
cv = KFold(n_splits=8, shuffle=True, random_state=40)
mae_scores = []
r2_scores = []

for train_index, val_index in cv.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    best_xgb_regressor.fit(X_train, y_train)
    y_pred_val = best_xgb_regressor.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    mae_scores.append(mae)
    r2_scores.append(r2)

avg_mae = np.mean(mae_scores) # mean error
avg_r2 = np.mean(r2_scores) # r2 

print(f"Average Mean Absolute Error (MAE) with 8-fold cross-validation: {avg_mae}")
print(f"Average R-squared (R2) with 8-fold cross-validation: {avg_r2}")

# Save the trained model
torch.save(best_xgb_regressor, 'xgboost_model.pth.tar')
