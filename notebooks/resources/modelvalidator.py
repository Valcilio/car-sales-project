from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn import metrics as m
from sklearn.model_selection import train_test_split

from resources.datatransformer import DataTransformer

class ModelValidator():

    def __init__(self, 
                model_name: str, 
                model: any, 
                X: pd.DataFrame = None, 
                y: pd.Series = None, 
                **kwargs):

        self.model_name = model_name
        self.model = model
        self.X = X
        self.y = y

    def regression_metrics(self, df_val: pd.DataFrame, n_yval: str = 'y', n_yhat: str = 'yhat', **kwargs):
        '''Calculate and return regression metrics'''

        df_val = df_val[df_val['y'] > 0]
        yval = df_val[n_yval]
        yhat = df_val[n_yhat]
        mae = m.mean_absolute_error(yval, yhat)
        mape = m.mean_absolute_percentage_error(yval, yhat)
        rmse = np.sqrt(m.mean_squared_error(yval, yhat))
        
        return pd.DataFrame({'Model Name': self.model_name,
                            'MAE': mae,
                            'MAPE': mape,
                            'RMSE': rmse}, index=[0])

    def kfolds_cross_val(self, y_scaler,
                         cv: int, verbose: bool = False, 
                         test_size: float = 0.2, **kwargs):
        '''Kfolds Cross-Validation for validate with a reality simulation the models' performance'''

        mae_list = []
        mape_list = []
        rmse_list = []

        for k in range(cv):
            if verbose:
                print( '\nKFold Number: {}'.format( k ) )

            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=test_size)

            self.model.fit(X_train, y_train) 
            yhat = self.model.predict(X_val)

            data_trans = DataTransformer(df=X_train)
            y_df = data_trans.reverse_concat_y(scaler=y_scaler, col_orig_name='leads', y_nt='log1p_leads', y_val=y_val, yhat=yhat)
            m_result = self.regression_metrics(y_df)

            # store performance of each kfold iteration
            mae_list.append(m_result['MAE'])
            mape_list.append(m_result['MAPE'])
            rmse_list.append(m_result['RMSE'])

        return pd.DataFrame({'Model Name': self.model_name,
                            'MAE CV': np.mean(mae_list),
                            'MAPE CV': np.mean(mape_list),
                            'RMSE CV': np.mean(rmse_list)}, 
                            index=[0])

    def shap_importance(self, **kwargs):
        '''Output a plot with shap values'''

        explainer = shap.Explainer(self.model, self.X)
        shap_values = explainer(self.X)
        shap.plots.beeswarm(shap_values)

    def plot_feature_importance(self, **kwargs):
        '''Plot feature importance values'''

        feat_imp = self._feat_imp_values()
        plt.subplots(figsize=(20,6))
        sns.barplot(x='feature_importance', y='feature', data=feat_imp, orient='h', color='royalblue')\
                    .set_title('Feature Importance');

    def _feat_imp_values(self, **kwargs):
        '''Feature importance values'''

        feat_imp = pd.DataFrame({'feature': self.X.columns,
                                'feature_importance': self.model.feature_importances_})\
                                .sort_values('feature_importance', ascending=False)\
                                .reset_index(drop=True)

        return feat_imp