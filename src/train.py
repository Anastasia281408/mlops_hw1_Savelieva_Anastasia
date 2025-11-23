import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-classification")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

X_train, y_train = train.drop("species", axis=1), train["species"]
X_test, y_test = test.drop("species", axis=1), test["species"]

model = LogisticRegression(
    C=params["train"]["C"],
    random_state=params["train"]["random_state"],
    max_iter=200
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="micro")

metrics = {"accuracy": acc, "f1_micro": f1}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

with mlflow.start_run():
    mlflow.log_params({
        "split_ratio": params["prepare"]["split_ratio"],
        "model_type": params["train"]["model_type"],
        "C": params["train"]["C"],
        "random_state": params["train"]["random_state"]
    })
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")
    mlflow.log_artifact("metrics.json")
