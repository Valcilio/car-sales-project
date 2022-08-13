import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest

from domain.connectors.datapreparator import DataPreparator

@pytest.fixture
def load_data():
    '''Load data for testing'''

    df_json = pd.read_csv('tests/test_data/test_df.csv').to_json()

    return {'data': df_json}

@pytest.fixture
def model_features():
    '''List of features for model'''

    return {'flg_aceita_troca':'int64', 
            'cod_tipo_pessoa':'int64',
            'min-max_best_offer':'float64', 
            'min-max_cliques_telefone*':'float64', 
            'min-max_views':'float64', 
            'min-max_ipva_dono':'float64'}

def test_dtypes(load_data, model_features):
    '''Test if the data types are returning correct after transformation'''

    df_pred = DataPreparator(json_df=load_data).preparate_data()
    cols_dtype = dict(df_pred.dtypes)

    assert cols_dtype == model_features
