import numpy as np
import pandas as pd
from   sklearn import preprocessing as pp

from domain.entities.announcement import Announcement

class Rescaler():

    def __init__(self, df: pd.DataFrame,
                best_offer_scaler: pp.MinMaxScaler,
                cliques_telefone_scaler: pp.MinMaxScaler,
                ipva_dono_scaler: pp.MinMaxScaler,
                views_scaler: pp.MinMaxScaler):
        
        self.df = Announcement(df).definy_dtypes()
        self.best_offer_scaler = best_offer_scaler
        self.cliques_telefone_scaler = cliques_telefone_scaler
        self.ipva_dono_scaler = ipva_dono_scaler
        self.views_scaler = views_scaler

    def run(self):
        '''Run all methods below for get data for modeling'''

        self._best_offer()
        self._cliques_telefone()
        self._ipva_dono()
        self._views()
        self._cod_tipo_pessoa()

        return self.df

    def _best_offer(self):
        '''Rescale the best_offer for a scale between 0 to 1'''

        self.df['min-max_best_offer'] = self.best_offer_scaler.transform(self.df[['best_offer']])

    def _cliques_telefone(self):
        '''Rescale the cliques_telefone* for a scale between 0 to 1'''

        self.df['min-max_cliques_telefone*'] = self.cliques_telefone_scaler.transform(self.df[['cliques_telefone*']])

    def _ipva_dono(self):
        '''Rescale the ipva_dono for a scale between 0 to 1'''

        self.df['min-max_ipva_dono'] = self.ipva_dono_scaler.transform(self.df[['ipva_dono']])

    def _views(self):
        '''Rescale the views for a scale between 0 to 1'''

        self.df['min-max_views'] = self.views_scaler.transform(self.df[['views']])

    def _cod_tipo_pessoa(self):
        '''Rescale the cod_tipo_pessoa for a scale between 0 to 1'''

        self.df['cod_tipo_pessoa'] = self.df['cod_tipo_pessoa'].apply(lambda x: 0 if x == 1 else 1)