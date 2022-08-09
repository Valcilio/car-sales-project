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

        self.df_pred = pd.DataFrame()
        self.df_pred.loc[0, 'cod_anuncio'] = self.json_df['cod_anuncio']
        self.df_pred.loc[0, 'cliques_telefone*'] = self.json_df['cliques_telefone*']
        self.df_pred.loc[0, 'flg_unico_dono'] = self.json_df['flg_unico_dono']
        self.df_pred.loc[0, 'flg_ipva_pago'] = self.json_df['flg_ipva_pago']
        self.df_pred.loc[0, 'flg_aceita_troca'] = self.json_df['flg_aceita_troca']
        self.df_pred.loc[0, 'views'] = self.json_df['views']
        self.df_pred.loc[0, 'cod_tipo_pessoa'] = self.json_df['cod_tipo_pessoa']
        self.df_pred.loc[0, 'prioridade'] = self.json_df['prioridade']
        self.df_pred.loc[0, 'flg_blindado'] = self.json_df['flg_blindado']
        self.df_pred.loc[0, 'flg_todas_revisoes_agenda_veiculo'] = self.json_df['flg_todas_revisoes_agenda_veiculo']

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