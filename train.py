import pandas as pd
import xgboost as xgb
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def make_data_set(raw_data_path, feature_df):
    data_file = os.path.join(raw_data_path, "train_dataset.parquet")
    data_df = pd.read_parquet(data_file)
    train_df, val_df = train_test_split(data_df, test_size=0.1, random_state=42)
    train_df = pd.merge(train_df, feature_df, on='ADDRESS', how='left')
    val_df = pd.merge(val_df, feature_df, on='ADDRESS', how='left')
    train_df = shuffle(train_df, random_state=42)
    train_x = train_df.drop(['ADDRESS', 'LABEL'], axis=1)
    train_y = train_df['LABEL'].astype(int)
    val_x = val_df.drop(['ADDRESS', 'LABEL'], axis=1)
    val_y = val_df['LABEL'].astype(int)
    return train_x, train_y, val_x, val_y


def get_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    res = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return res


def train_xgb(train_x, train_y, val_x, val_y, save_model_path):
    eval_set = [(val_x, val_y)]
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        early_stopping_rounds = 10
    )
    model.fit(
        train_x, train_y,
        eval_set=eval_set,
        verbose=True
    )
    best_iteration = model.best_iteration
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=best_iteration,
        use_label_encoder=False
    )
    model.fit(train_x, train_y)
    model.save_model(os.path.join(save_model_path, "best_model.json"))
    val_pred = model.predict(val_x)
    val_metrics = get_metrics(val_y, val_pred)
    for metric_name, value in val_metrics.items():
        print(f"val_metrics: {metric_name}: {value}")
    with open(os.path.join(save_model_path, "val_performance.json"), "w") as f:
        for metric_name, value in val_metrics.items():
            f.write(f"val_metrics: {metric_name}: {value}" + "\n")


def main():
    raw_data_path = "./data/raw_data/"
    feature_path = "./data/features/transactions_feature.parquet"
    save_model_path = "./saved_model/"
    feature_df = pd.read_parquet(feature_path)
    train_x, train_y, val_x, val_y = make_data_set(raw_data_path, feature_df)
    train_xgb(train_x, train_y, val_x, val_y, save_model_path)


if __name__ == "__main__":
    main()