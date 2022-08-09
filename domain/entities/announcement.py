import pandas as pd

class Announcement():

    def __init__(self, df: pd.DataFrame) -> None:
        
        self.df = df.copy()

    def definy_dtypes(self) -> pd.DataFrame:
        '''Definy the data types of the dataset for future transformations'''

        dtypes = {
                'cod_anuncio':'O',
                'cliques_telefone*':'float64',
                'flg_unico_dono':'O',
                'flg_ipva_pago':'O',
                'flg_aceita_troca':'int64',
                'views':'float64',
                'cod_tipo_pessoa':'int64',
                'prioridade':'O',
                'flg_blindado':'O',
                'flg_todas_revisoes_agenda_veiculo': 'O'
                }

        self.df = self.df.astype(dtypes)

        return self.df