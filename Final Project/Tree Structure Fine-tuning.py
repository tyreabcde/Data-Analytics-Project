import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

data_dir = Path("C:/Users/USER/Desktop/Course/Data Analytics/Final Project/")
l_train_s = pd.read_csv(data_dir / "data_analytics_datagame/light_train_source_labels.csv", index_col="user_id")
l_train_t = pd.read_csv(data_dir / "data_analytics_datagame/light_test_source_labels.csv", index_col="user_id")
slot_train_pr = pd.read_csv(data_dir / "slot_train_pr.csv", index_col="user_id")
slot_test_pr = pd.read_csv(data_dir / "slot_test_pr.csv", index_col="user_id")
slot_train_log = pd.read_csv(data_dir / "slot_train_log.csv", index_col="user_id")
slot_test_log = pd.read_csv(data_dir / "slot_test_log.csv", index_col="user_id")

X_train = pd.concat([l_train_s, slot_train_pr], axis=1)
y_train = l_train_t

model =  xgb.XGBClassifier(device='cuda', objective='binary:logistic', eval_metric='auc', scale_pos_weight=7)
param_dist = {
    "max_depth": [3,5,7],
    "min_child_weight": [1, 3, 5, 7, 9],
    "gamma": [0.1, 0.2, 0.3, 0.4, 0.5]
}
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    verbose=3,
    random_state=42,
)
random_search.fit(X_train, y_train)
print("Best parameters from RandomizedSearchCV:", random_search.best_params_)
best_params = random_search.best_params_

# 縮小範圍，進行 Grid Search 微調
param_grid = {
    "max_depth": [best_params["max_depth"] - 1, best_params["max_depth"], best_params["max_depth"] + 1],
    "min_child_weight": [best_params["min_child_weight"] - 1, best_params["min_child_weight"], best_params["min_child_weight"] + 1],
    "gamma": [best_params["gamma"] - 0.05, best_params["gamma"], best_params["gamma"] + 0.05]
}
param_grid = {k: [v for v in vals if v >= 0] for k, vals in param_grid.items()}
grid_search = GridSearchCV(
    model, param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    verbose=3,
)
grid_search.fit(X_train, y_train)

print("Best parameters from GridSearchCV:", grid_search.best_params_)
best_model = grid_search.best_estimator_
# 預測與評估
y_pred_train = best_model.predict_proba(X_train)
train_auc = roc_auc_score(y_train, y_pred_train)
print(f"Final Model Train AUC: {train_auc:.4f}")