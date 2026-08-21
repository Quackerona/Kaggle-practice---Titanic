import pandas as pd

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

print("TRAIN DATA: ", train_data.isna().sum())
print("TEST DATA: ", test_data.isna().sum())