"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm


def split_data(data_df):
    """Split a dataframe into training and validation datasets"""
    
    f = data_df.drop(['target', 'id'], axis = 1)
    lab = np.array(data_df['target'])
    f_train, f_valid, lab_train, lab_valid = train_test_split(f,
                                                              lab,
                                                              test_size=0.2,
                                                              random_state=0)

    t_data = lightgbm.Dataset(f_train, label=lab_train)
    v_data = lightgbm.Dataset(f_valid, label=lab_valid, free_raw_data=False)
    
    return (t_data, v_data)


def train_model(data, parameters):
    """Train a model with the given datasets and parameters"""
    # The object returned by split_data is a tuple.
    # Access train_data with data[0] and valid_data with data[1]
    
    model = lightgbm.train(parameters,
                           data[0],
                           valid_sets=data[1],
                           num_boost_round=500,
                           early_stopping_rounds=20)
    
    return model


def get_model_metrics(model, data):
    """Construct a dictionary of metrics for the model"""
    predictions = model.predict(data[1].data)
    y_pred = []
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
	        y_pred.append(1.0)
        else:
            y_pred.append(0.0)
    fpr, tpr, _ = metrics.roc_curve(data[1].label, predictions)
    model_metrics = {"auc": (metrics.auc(fpr, tpr)),
                     "f1": (metrics.f1_score(data[1].label, y_pred))}
    return model_metrics


def main():
    """This method invokes the training functions for development purposes"""
    
    # Read data from a file
    data_df = pd.read_csv('porto_seguro_safe_driver_prediction_input.csv')

    # Hard code the parameters for training the model
    parameters = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'sub_feature': 0.7,
        'num_leaves': 60,
        'min_data': 100,
        'min_hessian': 1,
        'verbose': 2
    }

    # Call the functions defined in this file
    data = split_data(data_df)
    model = train_model(data, parameters)
    model_metrics = get_model_metrics(model, data)
    
    # Print the resulting metrics for the model
    print(model_metrics)


if __name__ == '__main__':
    main()

