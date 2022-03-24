from copy import deepcopy
import numpy as np
import pandas as pd

import torch

from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from src.models.train_bnn_surv import train_cph_bnn

from sksurv.metrics import concordance_index_censored

import sys
sys.path.insert(1, '../auton-survival/')

import auton_survival.models.cph as ats
#import DeepCoxPh



def dataframe_to_datadicts(df, event_col='event', time_col='time'):
    # Extract the event and time columns as numpy arrays
    e = df[event_col].values.astype(np.int32)
    t = df[time_col].values.astype(np.float32)

    # Extract the patient's covariates as a numpy array
    x_df = df.drop([event_col, time_col], axis=1)
    x = x_df.values.astype(np.float32)

    # Return the deep surv dataframe
    return {
        'x': x,
        'e': e,
        't': t
    }

class eval_object():

    def __init__(self, train_data, test_data, dataset_name, time_column="time", event_column="event"):

        self.dataset_name = dataset_name
        self.event_column = event_column
        self.time_column = time_column

        self.label_columns = [event_column, time_column]
        self.feature_columns = list(set(list(train_data.columns)).difference(self.label_columns))

        self.train_data = train_data
        self.test_data = test_data

    def minmax_columns(self, feature_columns=None):
        if feature_columns == None:
            feature_columns = self.feature_columns

        scaler = MinMaxScaler()

        scaler.fit(self.train_data[feature_columns])
        self.train_data[feature_columns] = scaler.transform(self.train_data[feature_columns])

        scaler.fit(self.test_data[feature_columns])
        self.test_data[feature_columns] = scaler.transform(self.test_data[feature_columns])

    def standscale_columns(self, feature_columns=None):
        if feature_columns == None:
            feature_columns = self.feature_columns

        scaler = StandardScaler()

        scaler.fit(self.train_data[feature_columns])
        self.train_data[feature_columns] = scaler.transform(self.train_data[feature_columns])

        scaler.fit(self.test_data[feature_columns])
        self.test_data[feature_columns] = scaler.transform(self.test_data[feature_columns])

    def create_data_dicts(self):
        self.train_dict = dataframe_to_datadicts(self.train_data)
        self.test_dict = dataframe_to_datadicts(self.test_data)

    def fit_coxph(self, cox_estimator=None):
        if cox_estimator == None:
            cox_estimator = CoxPHSurvivalAnalysis()

        self.cox_estimator = self.fit_sklearn_estimator(cox_estimator)

    def get_coxph_metrics(self):
        self.coxph_metrics = self.evaluate_sklearn_method(self.cox_estimator, "coxph")
        return self.coxph_metrics

    def fit_rsf(self, rsf_estimator=None):
        if rsf_estimator == None:
            rsf_estimator = RandomSurvivalForest()
        self.rsf_estimator = self.fit_sklearn_estimator(rsf_estimator)

    def get_rsf_metrics(self):
        self.rsf_metrics = self.evaluate_sklearn_method(self.rsf_estimator, "rsf")
        return self.rsf_metrics

    def fit_deepcoxph(self, deepcoxmodel=None, epochs=50,ls = 0.001,bs = 100):
        if deepcoxmodel == None:
            deepcoxmodel = ats.DeepCoxPH(layers= [25,10,8])

        deepcoxmodel.fit(self.train_data[self.feature_columns].values,
                         self.train_data[self.time_column].values,
                         self.train_data[self.event_column].values,
                         iters=epochs,learning_rate = ls,batch_size = bs)

        self.deepcox_torchmodel = deepcoxmodel.torch_model[0]

    def get_deepcox_metrics(self):

        y_train_pred = self.deepcox_torchmodel(torch.from_numpy(self.train_data[self.feature_columns].values))
        y_train_pred = y_train_pred.detach().numpy()
        y_test_pred = self.deepcox_torchmodel(torch.from_numpy(self.test_data[self.feature_columns].values))
        y_test_pred = y_test_pred.detach().numpy()

        conc_train = concordance_index_censored(self.train_data[self.event_column].astype(bool).values,
                                                self.train_data[self.time_column].values, y_train_pred.ravel())[0]
        conc_test = concordance_index_censored(self.test_data[self.event_column].astype(bool).values,
                                               self.test_data[self.time_column].values, y_test_pred.ravel())[0]

        self.deepcox_metrics = pd.DataFrame({(conc_train, conc_test)}
                                            , columns=["train_conc", "test_conc"],
                                            index=[self.dataset_name + "_deepcox"])

        return self.deepcox_metrics

    def fit_sklearn_estimator(self, estimator):

        X = self.train_data[self.feature_columns].values
        y = np.array(self.train_data[self.label_columns].apply(tuple, axis=1).values,
                     dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

        estimator = estimator.fit(X, y)
        return estimator

    def evaluate_sklearn_method(self, sklearn_estimator, model_name):

        y_train_pred = sklearn_estimator.predict(self.train_data[self.feature_columns].values)
        y_test_pred = sklearn_estimator.predict(self.test_data[self.feature_columns].values)

        conc_train = concordance_index_censored(self.train_data[self.event_column].astype(bool).values,
                                                self.train_data[self.time_column].values, y_train_pred)[0]
        conc_test = concordance_index_censored(self.test_data[self.event_column].astype(bool).values,
                                               self.test_data[self.time_column].values, y_test_pred)[0]

        sklearn_metrics = pd.DataFrame({(conc_train, conc_test)}
                                       , columns=["train_conc", "test_conc"],
                                       index=[self.dataset_name + "_" + model_name])
        return sklearn_metrics

    def fit_coxbnn(self, torch_model=None, const_bnn_prior_parameters=None, epochs=50, patience=3, batchsize=256,
                   lr=1e-3):
        if torch_model == None:
            torch_model = deepcopy(self.deepcox_torchmodel)
        if const_bnn_prior_parameters == None:
            const_bnn_prior_parameters = {
                "prior_mu": 0.0,
                "prior_sigma": 1.0,
                "posterior_mu_init": 0.0,
                "posterior_rho_init": -3.0,
                "type": "Reparameterization",  # Flipout or Reparameterization
                "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
                "moped_delta": 0.5, }

        x = torch.from_numpy(self.train_data[self.feature_columns].values).float()
        t = torch.from_numpy(self.train_data[self.time_column].values).float()
        e = torch.from_numpy(self.train_data[self.event_column].values).float()

        dnn_to_bnn(torch_model, const_bnn_prior_parameters)

        self.deep_bnn = train_cph_bnn(torch_model, (x, t, e), epochs=epochs,
                                      patience=patience, batchsize=batchsize, lr=lr, debug=False,
                                      random_state=0, return_losses=False)

    def get_coxbnn_metrics(self):

        y_pred_train = self.deep_bnn(torch.from_numpy(self.train_data[self.feature_columns].values).float())
        y_pred_train = y_pred_train.data.cpu().numpy()
        y_pred_test = self.deep_bnn(torch.from_numpy(self.test_data[self.feature_columns].values).float())
        y_pred_test = y_pred_test.data.cpu().numpy()

        conc_train = concordance_index_censored(self.train_data[self.event_column].astype(bool).values
                                                , self.train_data[self.time_column].values, y_pred_train.ravel())[0]
        conc_test = concordance_index_censored(self.test_data[self.event_column].astype(bool).values
                                               , self.test_data[self.time_column].values, y_pred_test.ravel())[0]

        self.deepbnn_metrics = pd.DataFrame({(conc_train, conc_test)}
                                            , columns=["train_conc", "test_conc"],
                                            index=[self.dataset_name + "_deepbnn"])

        return self.deepbnn_metrics

    def fit_all_models(self):
        self.fit_coxph()
        self.fit_rsf()
        self.fit_deepcoxph()
        self.fit_coxbnn()

    def get_metrics(self):

        return pd.concat([
            self.get_coxph_metrics(),
            self.get_rsf_metrics(),
            self.get_deepcox_metrics(),
            self.get_coxbnn_metrics()])

