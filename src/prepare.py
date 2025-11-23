import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/raw/iris.csv")
species_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
df["species"] = df["species"].map(species_map)

train, test = train_test_split(
    df,
    test_size=1 - params["prepare"]["split_ratio"],
    random_state=params["prepare"]["random_state"],
    stratify=df["species"]
)

os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
