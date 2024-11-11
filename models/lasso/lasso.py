import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv('trainingSet.csv')

exclude_columns = ['Discharge_ID', 'Formula_dcg']  # not included in training
df = df.drop(exclude_columns, axis=1)

X = df.drop('Average_Voltage_(V)', axis=1)  # target_column
y = df['Average_Voltage_(V)']

imputer = SimpleImputer(strategy='mean')  # 'median', 'most_frequent'
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

lasso_regressor = Lasso(alpha=1e-10, 1e-08, 0.0001, 0.001, 0.01, 0.1, 10, 25, 30 ,35, 40 ,50 55) 


# validation
kf = KFold(n_splits=8, shuffle=True, random_state=42)
mse_scores = cross_val_score(lasso_regressor, X_scaled, y, scoring='neg_mean_squared_error', cv=kf)

mse_scores = -mse_scores
r2_scores = cross_val_score(lasso_regressor, X_scaled, y, scoring='r2', cv=kf)

# Print the results
print(f'Mean Squared Error Scores: {mse_scores}')
print(f'Mean Squared Error Mean: {mse_scores.mean()}')
print(f'R-squared Scores: {r2_scores}')
print(f'R-squared Mean: {r2_scores.mean()}')
