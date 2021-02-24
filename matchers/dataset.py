import pandas as pd
import numpy as np

from matchers import utils


def load_preprocess():
    df = pd.read_csv('../data/records25k_data.tsv', sep='\t', header=None)
    df.columns = ['name1', 'name2', 'co_occurrence', 'count1', 'count2']
    df.dropna(inplace=True)

    # Add padding
    df['name1'] = df['name1'].map(utils.add_padding)
    df['name2'] = df['name2'].map(utils.add_padding)

    df = df.groupby(['name1','name2']).agg({'co_occurrence': 'sum'}).groupby(level=0).apply(lambda x: x / x.sum()).reset_index()
    # df_name_matches = df.groupby('name1')['name2'].agg(list).reset_index()
    df_name_matches = df.groupby('name1').agg(list).reset_index()
    weighted_relevant_names = [[(n,w) for n,w in zip(ns,ws)] for ns,ws in zip(df_name_matches['name2'],df_name_matches['co_occurrence'])]
    input_names = df_name_matches['name1'].tolist()
    all_candidates = np.array(df['name2'].unique())

    # weighted_relevant_names = [[(name2, weight)]]
    # if you want just relevant names: [[name for name,weight in name_weights] for name_weights in weighted_relevant_names]
    return input_names, weighted_relevant_names, all_candidates
