import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv("data/train.csv")

train_data["FamilyCount"] = train_data["SibSp"] + train_data["Parch"]

survival_by_family_count = train_data.groupby("FamilyCount")["Survived"].mean()
survival_by_family_count.plot(ylabel="Survived")

plt.show()