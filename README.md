# Sybil Attack Address Detection Demo Pipeline

## Description
This is a machine learning demo example, convenient for beginners to learn and compete

## Specific steps
- Clone this repo. Install the dependencies from the requirements.txt file. If there are any other missing dependencies, please install them yourself.
- Create the following folders
    - ./data/
    - ./saved_model/
    - ./data/raw_data/
    - ./data/features/
- Place all the data files for the competition in the ./data/raw_data/ directory, included:
    - train_dataset.parquet
    - test_dataset.parquet
    - transactions.parquet
    - token_transfers.parquet
    - dex_swaps.parquet
- Run **`python feature_process.py`** to generate the feature file at ./data/features/transactions_feature.parquet.
- Run **`python train.py`** to train and save model.
- Run **`python inference.py`** to generate test dataset prediction result as pred.csv.
- Submit the pred.csv file to the website to view the accuracy of the test set.