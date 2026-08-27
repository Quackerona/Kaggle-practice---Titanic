import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv("data/train.csv")

bins = [0, 10, 25, 50, 100, 550]
train_data["FareGroup"] = pd.cut(train_data["Fare"], bins=bins)

survival_by_fare = train_data.groupby("FareGroup")["Survived"].mean()
survival_by_fare.plot(ylabel="Survived")

plt.show()