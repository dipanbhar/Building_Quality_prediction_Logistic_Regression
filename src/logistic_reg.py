import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Create logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'building_quality_model.pkl')
print("Model saved as 'building_quality_model.pkl'")
