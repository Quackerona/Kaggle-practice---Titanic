import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from optuna import create_study

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 700)
    max_depth = trial.suggest_int("max_depth", 5, 30, log=True)
    learning_rate = trial.suggest_float("learning_rate", 0.1, 0.6, log=True)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='binary:logistic'
    )

    y = train_data["Survived"]

    one_hot_features = ["Age", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[one_hot_features])
    X["Pclass"] = train_data["Pclass"]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    
    return scores.mean()

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

# Split data (Test).
test_data["Title"] = test_data.apply(
    lambda c: c["Name"].split(", ")[1].split(".")[0], 1
)

# Fill missing data (Test).
test_data.fillna({"Age": age_medians}, inplace=True)

# Training the model
model = XGBClassifier(
    n_estimators=135,
    max_depth=5,
    learning_rate=0.1117192728841936,
    objective='binary:logistic'
)

y = train_data["Survived"]

one_hot_features = ["Age", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[one_hot_features])
X["Pclass"] = train_data["Pclass"]

model.fit(X, y)

# Predict.
X_test = pd.get_dummies(test_data[one_hot_features])
X_test["Pclass"] = test_data["Pclass"]

result = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': result})
output.to_csv('submission.csv', index=False)

# # Finding the best parameters
# study = create_study(direction='maximize')
# study.optimize(objective, n_trials=200)

# print("Best Metric Value:", study.best_value)
# print("Best Hyperparameters:", study.best_params)
# print("Best Trial Number:", study.best_trial.number)