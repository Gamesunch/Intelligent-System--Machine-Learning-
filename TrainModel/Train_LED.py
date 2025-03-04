import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Life Expectancy Data.csv")

df.head()

df.tail()

# Clean column names
df.columns = df.columns.str.strip()

df.isnull().sum()

# Visualizing missing values before cleaning
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Values Before Cleaning")
plt.show()

# Encode categorical 'Status'
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})

# Drop rows where target variable is missing
df = df.dropna(subset=["Life expectancy"]).reset_index(drop=True)

# Handle missing values (impute with median)
df.fillna(df.median(numeric_only=True), inplace=True)

# Separate features and target variable
X = df.drop(columns=["Country", "Year", "Life expectancy"])  # Drop non-numeric columns
y = df["Life expectancy"]

# Visualizing missing values after cleaning
plt.figure(figsize=(10, 6))
sns.heatmap(X.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Values After Cleaning")
plt.show()

# Boxplot to visualize outliers before scaling
plt.figure(figsize=(12, 6))
sns.boxplot(data=X, orient='h')
plt.title("Feature Distributions Before Scaling")
plt.show()

# Scale features for SVM
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Boxplot to visualize feature distributions after scaling
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(X_scaled, columns=X.columns), orient='h')
plt.title("Feature Distributions After Scaling")
plt.show()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for SVM
svm_param_grid = {'C': [0.1, 1, 5, 10], 'epsilon': [0.001, 0.01, 0.1]}
svm_grid = GridSearchCV(SVR(kernel='rbf'), svm_param_grid, cv=5, scoring='r2', n_jobs=-1)
svm_grid.fit(X_train, y_train)
best_svm_model = svm_grid.best_estimator_

# Hyperparameter tuning for Random Forest
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf_model = rf_grid.best_estimator_

# Train models
best_svm_model.fit(X_train, y_train)
best_rf_model.fit(X_train, y_train)

# Save models
joblib.dump(best_svm_model, "svm_model.pkl")
joblib.dump(best_rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Predictions
y_pred_svm = best_svm_model.predict(X_test)
y_pred_rf = best_rf_model.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2 Score: {r2:.4f}\n")
    return mae, rmse, r2

svm_results = evaluate_model(y_test, y_pred_svm, "SVM")
rf_results = evaluate_model(y_test, y_pred_rf, "Random Forest")

# Save results to CSV
results_df = pd.DataFrame({
    "Model": ["SVM", "Random Forest"],
    "MAE": [svm_results[0], rf_results[0]],
    "RMSE": [svm_results[1], rf_results[1]],
    "R2 Score": [svm_results[2], rf_results[2]]
})
results_df.to_csv("model_performance.csv", index=False)

# Visualization
plt.figure(figsize=(8, 5))
sns.barplot(x=results_df["Model"], y=results_df["R2 Score"], palette="coolwarm")
plt.title("Model Comparison: R2 Score")
plt.ylabel("R2 Score")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.show()

print("Models saved and results saved to model_performance.csv")

