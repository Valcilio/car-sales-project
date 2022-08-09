import numpy as np
import pandas as pd
import pytest

from domain.entities.announcement import Announcement

@pytest.fixture
def correct_dtypes():
    '''Return the correct data types expected'''

    return {
            'cod_anuncio': np.dtype('O'),
            'cliques_telefone*': np.dtype('float64'),
            'flg_unico_dono': np.dtype('O'),
            'flg_ipva_pago': np.dtype('O'),
            'flg_aceita_troca': np.dtype('int64'),
            'views': np.dtype('float64'),
            'cod_tipo_pessoa': np.dtype('int64'),
            'prioridade': np.dtype('O'),
            'flg_blindado': np.dtype('O'),
            'flg_todas_revisoes_agenda_veiculo': np.dtype('O')
            }

def test_definy_dtypes(correct_dtypes):
    '''Test if the data types are returning correct after transformation'''

    df = pd.read_csv('tests/test_data/test_df.csv')
    df = Announcement(df=df).definy_dtypes()
    dtypes_df = dict(df.dtypes)

    assert correct_dtypes == dtypes_df
