import pandas as pd
import xgboost as xgb
import os

def read_test_feature(raw_data_path, feature_df):
    data_file = os.path.join(raw_data_path, "test_dataset.parquet")
    data_df = pd.read_parquet(data_file)
    test_df = pd.merge(data_df, feature_df, on='ADDRESS', how='left')
    test_x = test_df.drop(['ADDRESS'], axis=1)
    return data_df, test_x


def main():
    raw_data_path = "./data/raw_data/"
    feature_path = "./data/features/transactions_feature.parquet"
    feature_df = pd.read_parquet(feature_path)
    test_df, test_x = read_test_feature(raw_data_path, feature_df)
    save_model_path = "./saved_model/"
    best_model_path = os.path.join(save_model_path, "best_model.json")
    model = xgb.XGBClassifier()
    model.load_model(best_model_path)
    predictions = model.predict(test_x)
    test_df['PRED'] = predictions.astype(int)
    test_df.to_csv("pred.csv", index=False)
    
if __name__ == "__main__":
    main()

