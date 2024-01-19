from datetime import datetime as dt

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# parameters
seed = 42
test_size = 0.3


def split_df(df):
    X = df.drop("product02_bin", axis=1)
    y = df["product02_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test, X, y


def scoring(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # True pos + True negatives / all predictions
    print(f"Accuracy: {accuracy}")

    # True pos of all positive predicted
    print(f"Precision: {precision}")

    # True pos of true pos + false neg
    print(f"Recall: {recall}")

    # weighted accuracy and precision
    print(f"F1-Score: {f1}")
    return accuracy, precision, recall, f1


# read data
df_raw = pd.read_feather("data/processed/data-set.ftr")

features = [
    "income",
    "age",
    "var1",
    "lastVisit_year",
    "lastVisit_days",
    "product02_bin",
]
df = df_raw[features]

X_train, X_test, y_train, y_test, X, y = split_df(df)

model = "xgboost"
experiment_name = f"{model}_{dt.now().strftime('%Y%m%d-%H%M%S')}"

try:
    # creating a new experiment
    exp_id = mlflow.create_experiment(name=experiment_name)
except Exception as e:
    print(e)
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=exp_id):
    xgb_model = XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_y_pred = xgb_model.predict(X_test)

    # Scoring
    accuracy, precision, recall, f1 = scoring(y_test, xgb_y_pred)

    mlflow.sklearn.log_model(xgb_model, "xgb_model")
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    mlflow.log_metrics(metrics)
    mlflow.log_metrics(metrics)
