import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from optuna import create_study
from optuna.samplers import TPESampler
import shap

# Read data.
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Extract and filter Titles FIRST.
important_titles = ["Mr", "Miss", "Mrs"]
for df in [train_data, test_data]:
    df["Title"] = df["Name"].apply(lambda name: name.split(", ")[1].split(".")[0])
    df["Title"] = df["Title"].where(df["Title"].isin(important_titles), "Other")

# Extract and filter Decks.
important_decks = ["U", "E"]
for df in [train_data, test_data]:
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    df["Deck"] = df["Deck"].where(df["Deck"].isin(important_decks), "Other")

# Impute missing values cleanly using training medians.
age_medians = train_data.groupby("Title")["Age"].median()
global_age_median = train_data["Age"].median()
pclass_fare_medians = train_data.groupby("Pclass")["Fare"].median()

for df in [train_data, test_data]:
    df["Age"] = df["Age"].fillna(df["Title"].map(age_medians)).fillna(global_age_median)
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Fare"] = df["Fare"].fillna(df["Pclass"].map(pclass_fare_medians))
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Features and target.
y = train_data["Survived"]
features = ["FamilySize", "SibSp", "Parch", "Title", "Deck", "Embarked"]

X = pd.get_dummies(train_data[features])
X["Pclass"] = train_data["Pclass"]
X["Fare"] = train_data["Fare"]
X["Age"] = train_data["Age"]

X_test = pd.get_dummies(test_data[features])
X_test = X_test.reindex(columns=X.columns, fill_value=0)
X_test["Pclass"] = test_data["Pclass"]
X_test["Fare"] = test_data["Fare"]
X_test["Age"] = test_data["Age"]

# Optuna Objective.
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 300)
    max_depth = trial.suggest_int("max_depth", 3, 8)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=1,
    )

    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores.mean()

sampler = TPESampler(n_startup_trials=20, seed=1)
study = create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=200)

# # Train and Predict.
# model = RandomForestClassifier(
#     n_estimators=258,
#     max_depth=6,
#     min_samples_split=6,
#     min_samples_leaf=5,
#     random_state=1,
# )
# model.fit(X, y)

# prediction = model.predict(X_test)

# result = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": prediction})
# result.to_csv("submission.csv", index=False)

# explainer = shap.TreeExplainer(model)
# shap_values = explainer(X)

# total_features = len(X.columns)
# shap.plots.bar(shap_values[:, :, 1], max_display=total_features)