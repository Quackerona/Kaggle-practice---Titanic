# Purposefully over-commented since I need to remember this in the future.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold

# Reading data.
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Split data.
train_data["Title"] = train_data.apply(
    lambda c: c["Name"].split(", ")[1].split(".")[0], 1
)

# Fill missing data.
age_medians = train_data["Title"].map(train_data.groupby("Title")["Age"].median())
train_data.fillna({"Age": age_medians}, inplace=True)


# Training the model.
model = RandomForestClassifier(n_estimators=350, max_depth=5, random_state=1)

y = train_data["Survived"]

one_hot_features = ["Age", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[one_hot_features])
X["Pclass"] = train_data["Pclass"]

model.fit(X, y)

# Split data (Test).
test_data["Title"] = test_data.apply(
    lambda c: c["Name"].split(", ")[1].split(".")[0], 1
)

# Fill missing data (Test).
test_data.fillna({"Age": age_medians}, inplace=True)

# Predict.
X_test = pd.get_dummies(test_data[one_hot_features])
X_test["Pclass"] = test_data["Pclass"]

result = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': result})
output.to_csv('submission.csv', index=False)

# # Evaluate model performance
# cv = StratifiedKFold(n_splits=50, shuffle=True, random_state=1)
# scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

# print(f"Fold scores: {scores}")
# print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")