import os
import pandas as pd

def caculate_all_action_num(data_df, filter_df, columns=['FROM_ADDRESS', 'TO_ADDRESS'], final_name='TRANSACTION_ALL_ACTIONS_COUNT'):
    all_addresses = pd.concat([data_df['FROM_ADDRESS'], data_df['TO_ADDRESS']])
    address_counts = all_addresses.value_counts().reset_index()
    address_counts.columns = ['ADDRESS', final_name]
    result_df = address_counts[address_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_each_deposit_action_num(data_df, filter_df, column_name="TO_ADDRESS", final_name="TRANSACTION_DEPOSIT_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_each_deposit_action_num(data_df, filter_df, column_name="FROM_ADDRESS", final_name="TRANSACTION_WITHDRAWAL_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_unique_addresses(data_df, filter_df):
    from_to_unique = data_df.groupby('FROM_ADDRESS')['TO_ADDRESS'].nunique().reset_index()
    from_to_unique.columns = ['ADDRESS', 'TRANSACTION_UNIQUE_TO_INTERACTIONS']
    to_from_unique = data_df.groupby('TO_ADDRESS')['FROM_ADDRESS'].nunique().reset_index()
    to_from_unique.columns = ['ADDRESS', 'TRANSACTION_UNIQUE_FROM_INTERACTIONS']
    combined_interactions = pd.merge(from_to_unique, to_from_unique, on='ADDRESS', how='outer').fillna(0)
    result_df = combined_interactions[combined_interactions['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_max_nonce_per_contract(data_df, filter_df):
    melted_df = pd.melt(data_df, id_vars=['NONCE'], value_vars=['FROM_ADDRESS', 'TO_ADDRESS'], var_name='TYPE', value_name='ADDRESS')
    melted_df = melted_df.drop(columns='TYPE')
    max_nonce_per_address = melted_df.groupby('ADDRESS')['NONCE'].max().reset_index()
    result_df = max_nonce_per_address[max_nonce_per_address['ADDRESS'].isin(filter_df['ADDRESS'])]
    result_df = result_df.rename(columns={'NONCE': 'TRANSACTION_MAX_NONCE'})
    return result_df


#ETH平均转入value数
def calculate_deposit_average_value(data_df, filter_df):
    data_df['VALUE'] = pd.to_numeric(data_df['VALUE'], errors='coerce')
    average_values = data_df.groupby('TO_ADDRESS')['VALUE'].mean()
    deposit_average_values_df = average_values.reset_index(name='TRANSACTION_DEPOSIT_AVERAGE_VALUE')
    deposit_average_values_df.columns = ['ADDRESS', 'TRANSACTION_DEPOSIT_AVERAGE_VALUE']
    result_df = deposit_average_values_df[deposit_average_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


#ETH最大转入value数
def calculate_deposit_maximum_value(data_df, filter_df):
    data_df['VALUE'] = pd.to_numeric(data_df['VALUE'], errors='coerce')
    maximum_values = data_df.groupby('TO_ADDRESS')['VALUE'].max()
    deposit_maximum_values_df = maximum_values.reset_index(name='TRANSACTION_DEPOSIT_MAXIMUM_VALUE')
    deposit_maximum_values_df.columns = ['ADDRESS', 'TRANSACTION_DEPOSIT_MAXIMUM_VALUE']
    result_df = deposit_maximum_values_df[deposit_maximum_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


#ETH平均转出value数
def calculate_withdrawal_average_value(data_df, filter_df):
    data_df['VALUE'] = pd.to_numeric(data_df['VALUE'], errors='coerce')
    average_values = data_df.groupby('FROM_ADDRESS')['VALUE'].mean()
    withdrawal_average_values_df = average_values.reset_index(name='TRANSACTION_WITHDRAWAL_AVERAGE_VALUE')
    withdrawal_average_values_df.columns = ['ADDRESS', 'TRANSACTION_WITHDRAWAL_AVERAGE_VALUE']
    result_df = withdrawal_average_values_df[withdrawal_average_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

#ETH最大转出value数
def calculate_withdrawal_maximum_value(data_df, filter_df):
    data_df['VALUE'] = pd.to_numeric(data_df['VALUE'], errors='coerce')
    maximum_values = data_df.groupby('FROM_ADDRESS')['VALUE'].max()
    withdrawal_maximum_values_df = maximum_values.reset_index(name='TRANSACTION_WITHDRAWAL_MAXIMUM_VALUE')
    withdrawal_maximum_values_df.columns = ['ADDRESS', 'TRANSACTION_WITHDRAWAL_MAXIMUM_VALUE']
    result_df = withdrawal_maximum_values_df[withdrawal_maximum_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def make_transaction_features(transaction_raw_file, output_path, filter_df):
    data_df = pd.read_parquet(transaction_raw_file)

    all_action_counts_df = caculate_all_action_num(data_df, filter_df)
    print("all_action_counts_df done")
    deposit_action_num_df = calculate_each_deposit_action_num(data_df, filter_df)
    print("deposit_action_num_df done")
    each_deposit_action_num_df = calculate_each_deposit_action_num(data_df, filter_df)
    print("each_deposit_action_num_df done")
    unique_addresses_df = calculate_unique_addresses(data_df, filter_df)
    print("unique_addresses_df done")
    max_nonce_per_contract_df = calculate_max_nonce_per_contract(data_df, filter_df)
    print("max_nonce_per_contract_df done")
    deposit_average_value_df = calculate_deposit_average_value(data_df, filter_df)
    print("deposit_average_value_df done")
    deposit_maximum_value_df = calculate_deposit_maximum_value(data_df, filter_df)
    print("deposit_maximum_value_df done")
    withdrawal_average_value_df = calculate_withdrawal_average_value(data_df, filter_df)
    print("withdrawal_average_value_df done")
    withdrawal_maximum_value_df = calculate_withdrawal_maximum_value(data_df, filter_df)
    print("withdrawal_maximum_value_df done")

    final_df = filter_df.copy()
    dataframes = [
        all_action_counts_df,
        deposit_action_num_df,
        each_deposit_action_num_df,
        unique_addresses_df,
        max_nonce_per_contract_df,
        deposit_average_value_df,
        deposit_maximum_value_df,
        withdrawal_average_value_df,
        withdrawal_maximum_value_df
    ]

    for df in dataframes:
        final_df = pd.merge(final_df, df, on='ADDRESS', how='left')
    return final_df

