
# ---------------------------------------------------------
# Import Libraries
# ---------------------------------------------------------
import os
import joblib
import mlflow
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tourism-project-Package-Prediction-Experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

print("Loading datasets...")

Xtrain = pd.read_csv("hf://datasets/vidyaa2026/tourism_project/Xtrain.csv")
Xtest  = pd.read_csv("hf://datasets/vidyaa2026/tourism_project/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/vidyaa2026/tourism_project/ytrain.csv").squeeze()
ytest  = pd.read_csv("hf://datasets/vidyaa2026/tourism_project/ytest.csv").squeeze()

num_cols = Xtrain.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = Xtrain.select_dtypes(include=["object"]).columns.tolist()

preprocessor = make_column_transformer(
    ("passthrough", num_cols),
    (OneHotEncoder(handle_unknown="ignore"), cat_cols)
)

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    tree_method="hist"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgbclassifier", xgb)
])

param_dist = {
    "xgbclassifier__n_estimators":[100,200,300],
    "xgbclassifier__max_depth":[3,5,7],
    "xgbclassifier__learning_rate":[0.01,0.05,0.1],
    "xgbclassifier__subsample":[0.8,1.0],
    "xgbclassifier__colsample_bytree":[0.8,1.0],
    "xgbclassifier__min_child_weight":[1,3,5]
}

with mlflow.start_run():

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="recall",
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    search.fit(Xtrain, ytrain)

    best_model = search.best_estimator_

    print("\nBest Parameters")
    print(search.best_params_)
    print("\nBest CV Recall:", round(search.best_score_,4))

    ypred = best_model.predict(Xtest)
    yprob = best_model.predict_proba(Xtest)[:,1]

    metrics = {
        "accuracy": accuracy_score(ytest, ypred),
        "precision": precision_score(ytest, ypred),
        "recall": recall_score(ytest, ypred),
        "f1": f1_score(ytest, ypred),
        "roc_auc": roc_auc_score(ytest, yprob)
    }

    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")
        mlflow.log_metric(k,v)

    mlflow.log_params(search.best_params_)

    pd.DataFrame(confusion_matrix(ytest, ypred)).to_csv("confusion_matrix.csv", index=False)
    mlflow.log_artifact("confusion_matrix.csv")

    booster = best_model.named_steps["xgbclassifier"]
    feat_names = best_model.named_steps["preprocessor"].get_feature_names_out()

    fi = pd.DataFrame({
        "Feature": feat_names,
        "Importance": booster.feature_importances_
    }).sort_values("Importance", ascending=False)

    fi.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    model_name = "best_tourism_model.joblib"
    joblib.dump(best_model, model_name)
    mlflow.log_artifact(model_name)

repo_id="vidyaa2026/tourism-package-model"

try:
    api.repo_info(repo_id=repo_id, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_name,
    path_in_repo=model_name,
    repo_id=repo_id,
    repo_type="model"
)

print("XGBoost model uploaded successfully.")
