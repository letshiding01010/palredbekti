import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pyFTS.partitioners import Grid
from pyFTS.models import chen
from pyFTS.common import FLR
from pyFTS.common import Util
import numpy as np
from flask import Flask
from flask import render_template
from flask import request

data = pd.read_csv('4.csv')
data = data['4h'].values

fuzzy = Grid.GridPartitioner(data = data, npart = 11)
fuzzyfied = fuzzy.fuzzyfy(data, method = 'maximum', mode = 'sets')

model = chen.ConventionalFTS(partitioner = fuzzy)
model.fit(data)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.php')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    


    return render_template('index.php', prediction='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)