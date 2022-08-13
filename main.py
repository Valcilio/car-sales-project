import os
from flask import Flask, request, Response
import numpy as np
import pandas as pd

from domain.connectors.modelsrunner import ModelsRunner
from domain.connectors.datapreparator import DataPreparator

app = Flask(__name__)

@app.route('/framework_data', methods=['GET', 'POST'])
def car_leads_predict():
    '''Get json with instructions for get data, run pipeline and 
    reponse with json containing all data'''

    framework_json = request.get_json()

    if framework_json: 
        if isinstance(framework_json, dict): 
            pass

        pipeline = DataPreparator(json_df=framework_json)
        df_prepared = pipeline.preparate_data()

        model_run = ModelsRunner(df_prepared=df_prepared)
        class_pred = model_run.run_class_model()
        reg_pred = model_run.run_reg_model()

        df_response = pd.concat([class_pred, reg_pred], axis=1)
        df_response['number_of_leads'] = np.where(df_response['will_have_leads'] == 'no', np.nan, df_response['number_of_leads'])

        return df_response.to_json(orient='records', date_format='iso')
    else:
        return Response('{}', status='200', mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port, debug=True)