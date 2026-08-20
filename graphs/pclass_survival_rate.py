import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv("data/train.csv")

survival_by_status = train_data.groupby("Pclass")["Survived"].mean()
survival_by_status.plot(kind="bar", ylabel="Survived")

plt.show()