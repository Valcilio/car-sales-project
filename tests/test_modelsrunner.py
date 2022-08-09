import joblib
import pytest

from domain.connectors.datapreparator import DataPreparator
from domain.connectors.modelsrunner import ModelsRunner

@pytest.fixture
def load_data():
    '''Load data for testing'''

    return joblib.load(open('tests/test_data/dict_test.pkl', 'rb'))

@pytest.fixture
def model_return_dtype_reg():
    '''List of returned features from regression model'''

    return {'number_of_leads':'float64'}

@pytest.fixture
def model_return_dtype_class():
    '''List of returned features from classification model'''

    return {'will_have_leads':'O'}

@pytest.fixture
def get_pred_reg(load_data):
    '''Get prediction result of regression model'''

    df_reg = DataPreparator(json_df=load_data).preparate_data()
    pred_reg = ModelsRunner(df_prepared=df_reg).run_reg_model()

    return pred_reg
    
@pytest.fixture
def get_pred_class(load_data):
    '''Get prediction result of classification model'''

    df_class = DataPreparator(json_df=load_data).preparate_data()
    pred_class = ModelsRunner(df_prepared=df_class).run_class_model()

    return pred_class

def test_return_dtype_reg(model_return_dtype_reg ,get_pred_reg):
    '''Testing the dtype of the return from regression model'''

    col_pred_reg = dict(get_pred_reg.dtypes)

    assert col_pred_reg == model_return_dtype_reg

def test_return_dtype_class(model_return_dtype_class, get_pred_class):
    '''Testing the dtype of the return from classification model'''

    col_pred_class = dict(get_pred_class.dtypes)

    assert col_pred_class == model_return_dtype_class
