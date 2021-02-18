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

    df_name_matches = df.groupby('name1')['name2'].agg(list).reset_index()
    relevant_names = df_name_matches['name2'].tolist()
    input_names = df_name_matches['name1'].tolist()
    all_candidates = np.array(df['name2'].unique())

    return input_names, relevant_names, all_candidates
