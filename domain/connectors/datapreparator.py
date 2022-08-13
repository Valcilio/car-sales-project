import json
import joblib
import pandas as pd

from domain.use_cases.featurederivator import FeatureDerivator
from domain.use_cases.rescaler import Rescaler

class DataPreparator():

    def __init__(self, json_df: json):

        self.json_df = json_df

        self.best_offer_scaler = joblib.load(open('scalers/bo_scaler.pkl', 'rb'))
        self.cliques_telefone_scaler = joblib.load(open('scalers/ct_scaler.pkl', 'rb'))
        self.ipva_dono_scaler = joblib.load(open('scalers/ipvad_scaler.pkl', 'rb'))
        self.views_scaler = joblib.load(open('scalers/v_scaler.pkl', 'rb'))

    def preparate_data(self):
        '''Run all data preparation methods'''

        self._structure_json()
        self._run_feature_derivator()
        self._run_rescaler()
        self._filter_data()

        return self.df_pred

    def _structure_json(self):
        '''Structure JSON for prediction'''

        json_df = self.json_df['data']
        self.df_pred = pd.read_json(json_df)
        
    def _run_feature_derivator(self):
        '''Derivate the ipva_dono and best_offer variables'''

        feature_derivator = FeatureDerivator(df=self.df_pred)
        self.df_pred['ipva_dono'] = feature_derivator.ipva_dono()
        self.df_pred['best_offer'] = feature_derivator.best_offer()

    def _run_rescaler(self):
        '''Rescale all variables for modeling'''

        self.df_pred = Rescaler(df=self.df_pred,
                                best_offer_scaler=self.best_offer_scaler,
                                cliques_telefone_scaler=self.cliques_telefone_scaler,
                                ipva_dono_scaler=self.ipva_dono_scaler,
                                views_scaler=self.views_scaler).run()

    def _filter_data(self):
        '''Filter data for regression model'''

        keep_cols = ['flg_aceita_troca', 'cod_tipo_pessoa',
                    'min-max_best_offer', 'min-max_cliques_telefone*', 
                    'min-max_views', 'min-max_ipva_dono']

        self.df_pred = self.df_pred[keep_cols]