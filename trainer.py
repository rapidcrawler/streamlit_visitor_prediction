# train_model.py
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Sample data for fitting a Random Forest model
X_sample = np.random.rand(100, 10) * 100 + 1  # Add 1 to avoid log(0)
y_sample = np.dot(X_sample, np.array([50, 200, 300, 0.8, 1.2, 2.1, -0.5, 1.1, 0.7, -150])) + 10000

# Apply log transformation
X_sample_log = np.log(X_sample)
y_sample_log = np.log(y_sample)

# Fit a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_sample_log, y_sample_log)

# Save the model to a pickle file
artifact_path = './artifacts/'
with open(artifact_path+'random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print(artifact_path+'random_forest_model.pkl')
