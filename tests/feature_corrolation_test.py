import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

train_data = pd.read_csv("data/train.csv")

train_data["Deck"] = train_data["Cabin"].str[0]
train_data["Deck"] = train_data["Deck"].fillna("U")

train_data["Title"] = train_data.apply(
    lambda c: c["Name"].split(", ")[1].split(".")[0], 1
)

age_medians = train_data["Title"].map(train_data.groupby("Title")["Age"].median())
train_data.fillna({"Age": age_medians}, inplace=True)


def cramers_v(col_a, col_b):
    contingency = pd.crosstab(col_a, col_b)
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    r, k = contingency.shape
    return np.sqrt((chi2 / n) / (min(r, k) - 1))


def correlation_ratio(categories, values):
    categories = pd.Series(categories)
    values = pd.Series(values)
    overall_mean = values.mean()
    group_means = values.groupby(categories).mean()
    group_counts = values.groupby(categories).count()

    ss_between = (group_counts * (group_means - overall_mean) ** 2).sum()
    ss_total = ((values - overall_mean) ** 2).sum()

    return np.sqrt(ss_between / ss_total)

print("Social status")
print(f"(Pclass vs Deck) = {cramers_v(train_data['Pclass'], train_data['Deck']):.4f}")
print(f"(Pclass vs Fare) = {correlation_ratio(train_data['Pclass'], train_data['Fare']):.4f}")
print(f"(Deck vs Fare) = {correlation_ratio(train_data['Deck'], train_data['Fare']):.4f}")
print()

print("Role")
print(f"(Sex vs Title) = {cramers_v(train_data['Sex'], train_data['Title']):.4f}")
print(f"(Title vs Age) = {correlation_ratio(train_data['Title'], train_data['Age']):.4f}")