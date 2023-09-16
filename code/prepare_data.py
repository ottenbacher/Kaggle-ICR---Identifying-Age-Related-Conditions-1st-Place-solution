import pandas as pd
import numpy as np
import json

with open('../SETTINGS.json') as json_file:
    settings = json.load(json_file)

RAW_DATA_DIR = settings['RAW_DATA_DIR']
TRAIN_DATA_CLEAN_PATH = settings['TRAIN_DATA_CLEAN_PATH']
TEST_DATA_CLEAN_PATH = settings['TEST_DATA_CLEAN_PATH']

train_df = pd.read_csv(RAW_DATA_DIR + 'train.csv', index_col='Id')
test_df = pd.read_csv(RAW_DATA_DIR + 'test.csv', index_col='Id')
train_df['EJ'] = train_df['EJ'].replace({'A': 0, 'B': 1})
test_df['EJ'] = test_df['EJ'].replace({'A': 0, 'B': 1})
nan_fill = train_df.isna().any()
nan_fill *= train_df.min() - train_df.max()
nan_fill[nan_fill == 0] = train_df.median()
train_df = train_df.fillna(nan_fill)
test_df = test_df.fillna(nan_fill)

train_df.to_csv(TRAIN_DATA_CLEAN_PATH)
test_df.to_csv(TEST_DATA_CLEAN_PATH)
