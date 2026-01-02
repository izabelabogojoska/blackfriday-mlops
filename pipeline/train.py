import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

from features import build_features

def train_model():
    df = pd.read_csv("data.csv")

    # handle imbalance
    normal = df[df.abuse_label == 0]
    abuse = df[df.abuse_label == 1]

    normal_downsampled = resample(
        normal,
        replace=False,
        n_samples=len(abuse),
        random_state=42
    )

    df_balanced = pd.concat([normal_downsampled, abuse])

    X = build_features(df_balanced)
    y = df_balanced["abuse_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier()

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        mlflow.log_metric("auc", auc)
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="black_friday_abuse_model"
        )
