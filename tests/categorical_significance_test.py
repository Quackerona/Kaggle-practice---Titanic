import pandas as pd
from scipy.stats import chi2_contingency

train_data = pd.read_csv("data/train.csv")

train_data["Deck"] = train_data["Cabin"].str[0]
train_data["Deck"] = train_data["Deck"].fillna("U")

train_data["Title"] = train_data.apply(
    lambda c: c["Name"].split(", ")[1].split(".")[0], 1
)

train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1

for column in ["Embarked", "Deck", "Pclass", "Sex", "Title", "SibSp", "Parch", "FamilySize"]:
    contingency = pd.crosstab(train_data[column], train_data["Survived"])
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    print(f"{column}:")
    print("  Group sizes:", dict(train_data[column].value_counts()))
    print(f"  Chi2 = {chi2:.4f}, p-value = {p_value:.6f}")
    print("  Significant (p < 0.05):", p_value < 0.05)