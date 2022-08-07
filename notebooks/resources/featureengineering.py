import pandas as pd

class FeatureEngineering():

    def __init__(self, df: pd.DataFrame, **kwargs):
        '''This class is specific for this problem'''

        self.df = df

    def uf(self, **kwargs):
        
        df1 = self.df
        df1['uf'] = df1['uf_cidade'].apply(lambda x: x[0:2])

        return df1['uf']

    def decade(self, **kwargs):

        df1 = self.df
        df1['decade'] = df1['ano_modelo'].astype(int).apply(lambda x: 
                                                            '2000 <' if x < 2000 else
                                                            '2000 > | 2005 <' if (x > 2000) & (x < 2005) else
                                                            '2005 > | 2010 <' if (x > 2005) & (x < 2010) else
                                                            '2010 > | 2015 <' if (x > 2010) & (x < 2015) else
                                                            '2015 > | 2020 <' if (x > 2015) & (x < 2020) else
                                                            '2020 >')

        return df1['decade']

    def ipva_dono(self, **kwargs):

        df1 = self.df
        df1['unico_dono'] = df1['flg_unico_dono'].apply(lambda x: 'oneowner' if x == '1' else 'moreowners')
        df1['ipva_pago'] = df1['flg_ipva_pago'].apply(lambda x: 'paid' if x == '1.0' else 'no_paid')
        df1['ipva_dono'] = df1['ipva_pago'] + '_|_' + df1['unico_dono']
        
        return df1['ipva_dono']

    def sensors(self, **kwargs):

        df1 = self.df
        df1['schuva'] = df1['sensorchuva'].apply(lambda x: 'schuva' if x == 'S' else 'no_schuva')
        df1['srestacion'] = df1['sensorestacion'].apply(lambda x: 'restacion' if x == 'S' else 'no_rest')
        df1['sensors'] = df1['schuva'] + '_|_' + df1['srestacion']

        return df1['sensors']

    def eletr_car(self, **kwargs):

        df1 = self.df
        df1['el_trava'] = df1['travaeletr'].apply(lambda x: 'travaeletr' if x == 'S' else 'no_travaeletr')
        df1['el_vidro'] = df1['vidroseletr'].apply(lambda x: 'vidroseletr' if x == 'S' else 'no_vidroseletr')
        df1['eletr_car'] = df1['el_trava'] + '_|_' + df1['el_vidro']

        return df1['eletr_car']