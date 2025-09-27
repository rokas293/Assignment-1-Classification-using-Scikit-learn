import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
wildfire_training_data = pd.read_csv("wildfires_training.csv")
wildfire_test_data = pd.read_csv("wildfires_test.csv")

# Separate features
training_features = wildfire_training_data.drop("fire", axis=1)  
training_fire_labels = wildfire_training_data["fire"]            

test_features = wildfire_test_data.drop("fire", axis=1) 
test_fire_labels = wildfire_test_data["fire"]

# --- Model 1: Logistic Regression ---
wildfire_logistic_model = LogisticRegression(max_iter=1000)  # increased max_iter for convergence
wildfire_logistic_model.fit(training_features, training_fire_labels)

training_predictions_logistic = wildfire_logistic_model.predict(training_features)
test_predictions_logistic = wildfire_logistic_model.predict(test_features)

print("Logistic Regression:")
print("  Train Accuracy:", accuracy_score(training_fire_labels, training_predictions_logistic))
print("  Test Accuracy: ", accuracy_score(test_fire_labels, test_predictions_logistic))

