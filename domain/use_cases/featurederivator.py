import pandas as pd
from domain.entities.announcement import Announcement

class FeatureDerivator():

    def __init__(self, df: pd.DataFrame) -> None:

        self.df = Announcement(df).definy_dtypes()

    def ipva_dono(self):
        '''Feature who get together the IPVA paid feature and with only one owner feature'''

        self._str_ipva_dono()
        self.df['ipva_dono'] = self.df['ipva_dono'].apply(lambda x: 
                                                            0 if x == 'no_paid_|_moreowners' else 
                                                            1 if x == 'no_paid_|_oneowner' else
                                                            2 if x == 'paid_|_moreowners' else
                                                            3)
    
        return self.df['ipva_dono']

    def best_offer(self):
        '''Feature who indicates then rank of offer'''

        self._big_offer()
        first_offers = self._first_offers()
        second_offers = self._second_offers()
        third_offers = self._third_offers()
        self.df['best_offer'] = self.df['big_offer'].apply(lambda x: 
                                                    1 if x in first_offers else
                                                    2 if x in second_offers else
                                                    3 if x in third_offers else 
                                                    3)
        
        return self.df['best_offer'] 

    def _str_ipva_dono(self):
        '''Feature who get together the IPVA paid feature and with only one owner feature (STR)'''

        self.df = self.df
        self.df['unico_dono'] = self.df['flg_unico_dono'].apply(lambda x: 'oneowner' if x == '1' else 'moreowners')
        self.df['ipva_pago'] = self.df['flg_ipva_pago'].apply(lambda x: 'paid' if x == '1.0' else 'no_paid')
        self.df['ipva_dono'] = self.df['ipva_pago'] + '_|_' + self.df['unico_dono']

    def _big_offer(self):
        '''Feature who indicates if the car have the best offer possible'''

        self._text_rows_names()
        self.df['big_offer'] = (self.df['prioridade'] + '_|_' + self.df['flg_aceita_troca'] + '_|_' + self.df['flg_blindado'] + '_|_' + self.df['flg_todas_revisoes_agenda_veiculo'])

    def _text_rows_names(self):
        '''Passing rows' names to text'''

        self.df['prioridade'] = self.df['prioridade'].apply(lambda x: 'prio' if x == '2' else 'no_prio')
        self.df['flg_aceita_troca'] = self.df['flg_aceita_troca'].apply(lambda x: 'troca' if x == 1 else 'no_troca')
        self.df['flg_blindado'] = self.df['flg_blindado'].apply(lambda x: 'blind' if x == '1' else 'no_blind')
        self.df['flg_todas_revisoes_agenda_veiculo'] = self.df['flg_todas_revisoes_agenda_veiculo'].apply(lambda x: 'all_rev' if x == '1' else 'no_rev')

    def _first_offers(self):
        '''Best offers who call more attention from customers'''

        return ['no_prio_|_troca_|_blind_|_all_rev', 'no_prio_|_troca_|_blind_|_no_rev',
               'prio_|_no_troca_|_blind_|_all_rev', 'prio_|_troca_|_blind_|_all_rev']

    def _second_offers(self):
        '''Second best offers who call more attention from customers'''

        return ['no_prio_|_no_troca_|_blind_|_all_rev', 'no_prio_|_troca_|_no_blind_|_all_rev',
                'no_prio_|_troca_|_no_blind_|_no_rev', 'prio_|_no_troca_|_blind_|_no_rev',
                'prio_|_no_troca_|_no_blind_|_all_rev', 'prio_|_troca_|_blind_|_no_rev',
                'prio_|_troca_|_no_blind_|_all_rev', 'prio_|_troca_|_no_blind_|_no_rev']

    def _third_offers(self):
        '''Worst offers who call less attention from customers'''

        return ['no_prio_|_no_troca_|_blind_|_no_rev', 'no_prio_|_no_troca_|_no_blind_|_all_rev',
                'no_prio_|_no_troca_|_no_blind_|_no_rev', 'prio_|_no_troca_|_no_blind_|_no_rev']