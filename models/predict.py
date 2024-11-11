import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import torch

trained_model = torch.load('xgboost_model.pth.tar')

new_data = pd.read_csv('delete.csv')

X_new = new_data.drop(columns=['Formation_Energy_Per_Atom_dcg', 'Battery_ID','Material_IDs',	'Gravimetric_Capacity_(mAh/g)', 'Average_Voltage_(V)',	'Volumetric_Capacity (Ah/L)'])
y_new = new_data['Formation_Energy_Per_Atom_dcg']

scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

y_pred_new = trained_model.predict(X_new_scaled)

mae_new = mean_absolute_error(y_new, y_pred_new)
r2_new = r2_score(y_new, y_pred_new)

print(f"Mean Absolute Error (MAE) on the new data: {mae_new}")
print(f"R-squared (R2) on the new data: {r2_new}")
print(y_pred_new)
