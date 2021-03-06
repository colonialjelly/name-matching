import pandas as pd
import numpy as np

from matchers import utils

DATASET_NAME = 'records25k_data'


def load_split_init(test_size=0.1):
    df = pd.read_csv(f'../data/{DATASET_NAME}.tsv', sep='\t', header=None)
    df.columns = ['name1', 'name2', 'co_occurrence', 'count1', 'count2']
    df.dropna(inplace=True)

    # Split train test
    df_train, df_test = train_test_split(df, test_size=test_size)

    # Persist splits on disk
    df_train.to_csv(f'../data/{DATASET_NAME}_train.csv')
    df_test.to_csv(f'../data/{DATASET_NAME}_test.csv')


def load_process_from_disk():
    df_train = pd.read_csv(f'../data/{DATASET_NAME}_train.csv')
    df_test = pd.read_csv(f'../data/{DATASET_NAME}_test.csv')
    input_names_train, weighted_relevant_names_train, all_candidates_train = process(df_train.copy())
    input_names_test, weighted_relevant_names_test, all_candidates_test = process(df_test.copy())

    return (input_names_train, weighted_relevant_names_train, all_candidates_train), \
           (input_names_test, weighted_relevant_names_test, all_candidates_test)


def process(df):
    # Add padding
    df.loc[:, 'name1'] = df.loc[:, 'name1'].map(utils.add_padding)
    df.loc[:, 'name2'] = df.loc[:, 'name2'].map(utils.add_padding)

    df = df.groupby(['name1', 'name2']) \
        .agg({'co_occurrence': 'sum'}) \
        .groupby(level=0) \
        .apply(lambda x: x / x.sum()) \
        .reset_index()

    df_name_matches = df.groupby('name1').agg(list).reset_index()
    weighted_relevant_names = [[(n, w) for n, w in zip(ns, ws)] for ns, ws in
                               zip(df_name_matches['name2'], df_name_matches['co_occurrence'])]
    input_names = df_name_matches['name1'].tolist()
    all_candidates = np.array(df['name2'].unique())

    # if you want just relevant names:
    # [[name for name,weight in name_weights] for name_weights in weighted_relevant_names]
    return input_names, weighted_relevant_names, all_candidates


def train_test_split(df, test_size=0.1):
    msk = np.random.uniform(0, 1, len(df)) < 1 - test_size
    df_train = df[msk].copy()
    df_test = df[~msk].copy()

    # Find record names that are both in train and test
    train_names = list(df_train['name2'].unique())
    msk = df_test['name2'].isin(train_names)
    df_duplicated = df_test[msk].copy()

    # Remove duplicated names from test and add it to train
    df_test.drop(df_test[msk].index, inplace=True)
    df_train = pd.concat([df_train, df_duplicated], axis=0)

    assert not len(set(df_train['name2'].tolist()).intersection(set(df_test['name2'].tolist())))

    return df_train, df_test
