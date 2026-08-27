import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv("data/train.csv")

train_data["Deck"] = train_data["Cabin"].str[0]
train_data["Deck"] = train_data["Deck"].fillna("U")

survival_by_deck = train_data.groupby("Deck")["Survived"].mean()
survival_by_deck.plot(kind="bar", ylabel="Survived")

plt.show()