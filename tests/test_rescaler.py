import joblib
import pandas as pd
import pytest
import warnings

from domain.use_cases.featurederivator import FeatureDerivator
from domain.use_cases.rescaler import Rescaler

@pytest.fixture
def load_data():
    '''Load data for testing'''

    return pd.read_csv('tests/test_data/test_df.csv')

@pytest.fixture
def add_cols(load_data):
    '''Derivating new cols for the models'''

    df = load_data
    feature_derivator = FeatureDerivator(df=load_data)
    df['ipva_dono'] = feature_derivator.ipva_dono()
    df['best_offer'] = feature_derivator.best_offer()

    return df

@pytest.fixture
def load_scalers():
    '''Load scalers for transform data'''

    warnings.filterwarnings(action='ignore')
    best_offer_scaler = joblib.load(open('scalers/bo_scaler.pkl', 'rb'))
    cliques_telefone_scaler = joblib.load(open('scalers/ct_scaler.pkl', 'rb'))
    ipva_dono = joblib.load(open('scalers/ipvad_scaler.pkl', 'rb'))
    views = joblib.load(open('scalers/v_scaler.pkl', 'rb'))
    
    return best_offer_scaler, cliques_telefone_scaler, ipva_dono, views

@pytest.fixture
def rescaled_features():
    '''List of rescaled features for testing'''

    return ['min-max_best_offer', 'min-max_cliques_telefone*', 
            'min-max_ipva_dono', 'min-max_views', 'flg_aceita_troca',
            'cod_tipo_pessoa']

@pytest.fixture
def run_rescaler(add_cols, load_scalers):
    '''Run rescaler for get data for testing'''

    warnings.filterwarnings(action='ignore')
    rescaler = Rescaler(df=add_cols, 
                        best_offer_scaler=load_scalers[0],
                        cliques_telefone_scaler=load_scalers[1],
                        ipva_dono_scaler=load_scalers[2],
                        views_scaler=load_scalers[3])

    df = rescaler.run() 

    return df

def test_rescalling(run_rescaler, rescaled_features):
    '''Test if the data types are returning correct after transformation'''

    for feature in rescaled_features:
        print(feature)
        print(run_rescaler['cod_tipo_pessoa'].unique())
        assert ((run_rescaler[feature].max() <= 1) & (run_rescaler[feature].min() >= 0))
