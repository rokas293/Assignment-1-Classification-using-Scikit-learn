import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
train_df = pd.read_csv("wildfires_training.csv")
test_df = pd.read_csv("wildfires_test.csv")

# Separate features (X) and target (Y)
X_train = train_df.drop("fire", axis=1)
y_train = train_df["fire"]

X_test = test_df.drop("fire", axis=1)
y_test = test_df["fire"]

