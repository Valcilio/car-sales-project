import numpy as np
import pandas as pd
import pytest

from domain.use_cases.featurederivator import FeatureDerivator

@pytest.fixture
def load_data():
    '''Load data for testing'''

    return pd.read_csv('tests/test_data/test_df.csv')

@pytest.fixture
def correct_dtypes():
    '''Return the correct data types expected'''

    return {
            'ipva_dono': np.dtype('int64'),
            'best_offer': np.dtype('int64')
            }

def test_dtypes(load_data, correct_dtypes):
    '''Test if the data types are returning correct after transformation'''

    df = load_data
    feature_derivator = FeatureDerivator(df=df)
    df['ipva_dono'] = feature_derivator.ipva_dono()
    df['best_offer'] = feature_derivator.best_offer()
    df = df[['ipva_dono', 'best_offer']]
    dtypes_df = dict(df.dtypes)

    assert correct_dtypes == dtypes_df
