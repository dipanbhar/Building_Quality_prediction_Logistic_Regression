import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load dataset (replace with actual path or DataFrame)
data = pd.read_csv('../data/building_quality.csv')

# Display first few rows of data
print("Data Preview:")
print(data.head())

# Assume the dataset has features and a 'quality' column where 1 = good, 0 = bad
X = data.drop('quality', axis=1)

y = data['quality']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression(max_iter=10000, random_state=42))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)


# Make predictions
y_pred_train_log_reg = pipeline.predict(X_train)
y_pred_test_log_reg = pipeline.predict(X_test)

# Evaluate the model
# Calculate the training and testing accuracy
training_accuracy = accuracy_score(y_train, y_pred_train_log_reg)
testing_accuracy = accuracy_score(y_test, y_pred_test_log_reg)

print("Logistic Regression with pipeline")
print(f"Training Accuracy: {training_accuracy}")
print(f"Testing Accuracy: {testing_accuracy}")

# Save model
joblib.dump(pipeline, 'building_quality_pipeline.pkl')
print("Pipeline saved as 'building_quality_pipeline.pkl'")
