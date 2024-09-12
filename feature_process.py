import os
import pandas as pd
import FeatureEngineering as fe

def get_all_candidates(file_path):
    train_datafile = os.path.join(file_path, "train_dataset.parquet")
    test_datafile = os.path.join(file_path, "test_dataset.parquet")
    train_df = pd.read_parquet(train_datafile)
    test_df = pd.read_parquet(test_datafile)
    train_addresses = train_df['ADDRESS']
    test_addresses = test_df['ADDRESS']
    all_addresses = pd.concat([train_addresses, test_addresses], ignore_index=True)
    all_addresses_df = pd.DataFrame(all_addresses, columns=['ADDRESS'])
    return all_addresses_df

    
if __name__ == "__main__":
    file_path = "./data/raw_data/"
    output_path = "./data/features/"
    transaction_raw_file = os.path.join(file_path, "transactions.parquet")
    token_transfer_raw_file = os.path.join(file_path, "token_transfers.parquet")
    dex_swaps_raw_file = os.path.join(file_path, "dex_swaps.parquet")
    address_df = get_all_candidates(file_path)
    transaction_feature_df = fe.make_transaction_features(transaction_raw_file, output_path, address_df)
    transaction_feature_df.fillna(0, inplace=True)
    output_file_name = os.path.join(output_path, "transactions_feature.parquet")
    transaction_feature_df.to_parquet(output_file_name)



    