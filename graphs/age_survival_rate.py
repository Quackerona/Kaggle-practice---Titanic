import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv("data/train.csv")

bins = [0, 12, 18, 30, 50, 80]
train_data["AgeGroup"] = pd.cut(train_data["Age"], bins=bins)

survival_by_group = train_data.groupby("AgeGroup")["Survived"].mean()
survival_by_group.plot(ylabel="Survived")

plt.show()