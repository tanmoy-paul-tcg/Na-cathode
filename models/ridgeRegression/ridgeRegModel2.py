import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

data = pd.read_csv('trainingSet.csv')

columns_to_omit = ["Discharge_ID",'Formula_dcg']  #will not take part in training.


X = data.drop(['Average_Voltage_(V)'] + columns_to_omit, axis=1)
y = data['Average_Voltage_(V)']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ridge_model = Ridge(alpha=20)  


kf = KFold(n_splits=8, shuffle=True, random_state=42)

rmse_list = []
r2_list = []

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    ridge_model.fit(X_train, y_train)

    y_pred = ridge_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rmse_list.append(rmse)
    r2_list.append(r2)


print(f'Average RMSE: {np.mean(rmse_list)}')
print(f'Average R-squared: {np.mean(r2_list)}')

model_filename = 'ridge_model.pth.tar'
joblib.dump(ridge_model, model_filename)

scaler_filename = 'scaler_model.pkl'
joblib.dump(scaler, scaler_filename)

print(f'Model and scaler saved to {model_filename} and {scaler_filename} respectively.')
