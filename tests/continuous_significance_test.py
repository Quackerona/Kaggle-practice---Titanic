import pandas as pd
from scipy.stats import ttest_ind

train_data = pd.read_csv("data/train.csv")

for column in ["Fare", "Age"]:
    survived = train_data[train_data["Survived"] == 1][column].dropna()
    not_survived = train_data[train_data["Survived"] == 0][column].dropna()

    t_stat, p_value = ttest_ind(survived, not_survived, equal_var=False)

    print(f"{column}:")
    print(f"  Mean (Survived) = {survived.mean():.2f}, Mean (Not survived) = {not_survived.mean():.2f}")
    print(f"  t = {t_stat:.4f}, p-value = {p_value:.6f}")
    print("  Significant (p < 0.05):", p_value < 0.05)