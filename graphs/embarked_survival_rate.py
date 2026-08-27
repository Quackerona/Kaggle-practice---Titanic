import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv("data/train.csv")

survival_by_embarked = train_data.groupby("Embarked")["Survived"].mean()
survival_by_embarked.plot(kind="bar", ylabel="Survived")

plt.show()