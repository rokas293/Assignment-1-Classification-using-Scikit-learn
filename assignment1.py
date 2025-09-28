import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
wildfire_training_data = pd.read_csv("wildfires_training.csv")
wildfire_test_data = pd.read_csv("wildfires_test.csv")

# Separate features
training_features = wildfire_training_data.drop("fire", axis=1)  
training_fire_labels = wildfire_training_data["fire"]            

test_features = wildfire_test_data.drop("fire", axis=1) 
test_fire_labels = wildfire_test_data["fire"]


# Data statistics before training


print("Dataset Overview Before Training")

print(f"Training samples: {len(training_features)}")
print(f"Test samples: {len(test_features)}")
print(f"Number of features: {len(training_features.columns)}")
print(f"Features: {list(training_features.columns)}")

print(f"\nClass distribution in training data:")
class_counts = training_fire_labels.value_counts()
for class_name, count in class_counts.items():
    percentage = (count / len(training_fire_labels)) * 100
    print(f"  {class_name}: {count} samples ({percentage:.1f}%)\n")


# --- Model 1: Logistic Regression ---
# Create the model using default parameters
wildfire_logistic_model = LogisticRegression(max_iter=1000)  # increased max_iter for convergence
# Train the model using the training data
wildfire_logistic_model.fit(training_features, training_fire_labels)

# Make predictions on both training and test datasets
training_predictions_logistic = wildfire_logistic_model.predict(training_features)
test_predictions_logistic = wildfire_logistic_model.predict(test_features)

# Print results
print("Logistic Regression:")
print("  Train Accuracy:", accuracy_score(training_fire_labels, training_predictions_logistic))
print("  Test Accuracy: ", accuracy_score(test_fire_labels, test_predictions_logistic))

# --- Model 2: Random Forest ---
# Create the Random Forest model but with default parameters
wildfire_forest_model = RandomForestClassifier(random_state=42)

# Train the random forest model using the training data
wildfire_forest_model.fit(training_features, training_fire_labels)

# Make predictions on both training and test datasets
training_predictions_forest = wildfire_forest_model.predict(training_features)
test_predictions_forest = wildfire_forest_model.predict(test_features)

# Print results
print("\nRandom Forest:")
print("  Train Accuracy:", accuracy_score(training_fire_labels, training_predictions_forest))
print("  Test Accuracy: ", accuracy_score(test_fire_labels, test_predictions_forest))


# Logistic Regression hyperparameter tuning
print("\n1. Logistic Regression hyperparameter tuning\n")

# Test different C values to find the best one
C_values = [0.01, 0.1, 1, 10, 100]
solver = 'lbfgs'  # Using lbfgs solver for this tuning

print(f"Testing different C values with {solver} solver...")

best_accuracy = 0
best_C_Variable = None
results_logistic = []

# Manually search through the different C values
for C in C_values:
    # Create the model with the specific c value and solver
    tuned_logistics_model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
    
    # Train the model
    tuned_logistics_model.fit(training_features, training_fire_labels)

    # Make predictions on the test set
    test_predictions_tuned = tuned_logistics_model.predict(test_features)
    test_accuracy = accuracy_score(test_fire_labels, test_predictions_tuned)
    
    # Store results
    results_logistic.append({
        'C': C,
        'test_accuracy': test_accuracy
    })
    
    print(f"C={C:6}: Test Accuracy = {test_accuracy:.3f}")
    
    # Track the best performance between all C values
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_C_Variable = C

# Print best results for Logistic Regression and comparison to default
print(f"\nBest Logistic Regression parameters:")
print(f"  C = {best_C_Variable}, solver = {solver}")
print(f"Best test accuracy: {best_accuracy:.3f}")
print(f"Default test accuracy: {accuracy_score(test_fire_labels, test_predictions_logistic):.3f}")
print(f"Improvement: {best_accuracy - accuracy_score(test_fire_labels, test_predictions_logistic):.3f}")


# Random Forest hyperparameter tuning

print(f"\n2. Random Forest hyperparameter tuning\n")

# 2 hyperparameters to tune: n_estimators and max_depth
n_estimators_values = [10, 50, 100, 200] # Number of trees in the forest
max_depth_values = [3, 5, 10, None] # Maximum depth of each tree

print("Testing different combinations of n_estimators and max_depth...")

best_accuracy_randomforest = 0
best_params_randomforest = {}
results_randomforest = []

# Manually search through combinations of n_estimators and max_depth
for n_est in n_estimators_values:
    for max_d in max_depth_values:
        # Create model with the specific hyperparameters
        tuned_randomforest_model = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            random_state=42
        )
        
        # Train the model with training data
        tuned_randomforest_model.fit(training_features, training_fire_labels)
        
        # Make predictions on test set, so new data is used
        test_predictions_tuned = tuned_randomforest_model.predict(test_features)
        test_accuracy = accuracy_score(test_fire_labels, test_predictions_tuned)
        
        # Store results
        results_randomforest.append({
            'n_estimators': n_est,
            'max_depth': max_d,
            'test_accuracy': test_accuracy
        })
        
        depth_string = str(max_d) if max_d is not None else "None"
        print(f"num_trees={n_est:3}, max_depth={depth_string:4}: Test Accuracy = {test_accuracy:.3f}")

        # Track the best performance between all combinations
        if test_accuracy > best_accuracy_randomforest:
            best_accuracy_randomforest = test_accuracy
            best_params_randomforest = {'n_estimators': n_est, 'max_depth': max_d}

# Print best results for Random Forest and comparison to default
print(f"\nBest Random Forest parameters: {best_params_randomforest}")
print(f"Best test accuracy: {best_accuracy_randomforest:.3f}")
print(f"Default test accuracy: {accuracy_score(test_fire_labels, test_predictions_forest):.3f}")
print(f"Improvement: {best_accuracy_randomforest - accuracy_score(test_fire_labels, test_predictions_forest):.3f}")


# Final comparison summary


print("\nResults Summary")

# Compare best models from both algorithms
# Logistic Regression
print("\nLogistic Regression:")
print(f"  Default hyperparameters: Test Accuracy = {accuracy_score(test_fire_labels, test_predictions_logistic):.3f}")
print(f"  Best tuned hyperparameters: Test Accuracy = {best_accuracy:.3f}")
print(f"  Best parameters: C={best_C_Variable}, solver={solver}")
print(f"  Improvement from tuning: {best_accuracy - accuracy_score(test_fire_labels, test_predictions_logistic):.3f}")

# Random Forest
print("\nRandom Forest:")
print(f"  Default hyperparameters: Test Accuracy = {accuracy_score(test_fire_labels, test_predictions_forest):.3f}")
print(f"  Best tuned hyperparameters: Test Accuracy = {best_accuracy_randomforest:.3f}")
print(f"  Best parameters: {best_params_randomforest}")
print(f"  Improvement from tuning: {best_accuracy_randomforest - accuracy_score(test_fire_labels, test_predictions_forest):.3f}")

# Overall best model
print(f"\nOverall best model:")
if best_accuracy > best_accuracy_randomforest:
    print(f"  Logistic Regression with C={best_C_Variable}, solver={solver}")
    print(f"  Test Accuracy: {best_accuracy:.3f}")
else:
    print(f"  Random Forest with {best_params_randomforest}")
    print(f"  Test Accuracy: {best_accuracy_randomforest:.3f}")

# Comparison graph
print(f"\nGenerating comparison...")

models = ['Logistic Default', 'Logistic Best', 'RandomF Default', 'RandomF Best']
scores = [
    accuracy_score(test_fire_labels, test_predictions_logistic),
    best_accuracy,
    accuracy_score(test_fire_labels, test_predictions_forest),
    best_accuracy_randomforest
]

plt.figure(figsize=(8, 5))
plt.bar(models, scores, color=['lightblue', 'blue', 'lightcoral', 'red'])
plt.ylabel('Test Accuracy')
plt.title('Algorithm Comparison')
plt.ylim(0.8, 1.0)
plt.show()
