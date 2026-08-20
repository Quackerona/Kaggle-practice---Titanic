import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv("data/train.csv")

gender_survived = train_data.groupby("Sex")["Survived"].mean()
gender_survived.plot(kind="pie", ylabel="Survived", autopct="%1.1f%%")

plt.show()