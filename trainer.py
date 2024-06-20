# train_model.py
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Sample data for fitting a simple linear regression model
X_sample = np.random.rand(100, 13) * 100 + 1  # Add 1 to avoid log(0)
y_sample = np.dot(X_sample, np.array([50, 200, 300, 0.8, 1.2, 2.1, -0.5, 1.1, 0.7, -150, 0.5, 100, 200])) + 10000

# Apply log transformation
X_sample_log = np.log(X_sample)
y_sample_log = np.log(y_sample)

# Fit a simple linear regression model
model = LinearRegression()
model.fit(X_sample_log, y_sample_log)

# Save the model to a pickle file
artifact_path = './artifacts/'
with open(artifact_path+'linear_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print(artifact_path+'linear_model.pkl')
