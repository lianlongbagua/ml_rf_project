from datetime import datetime

import pandas as pd
import numpy as np
import talib

try:
    from sklearn.impute import IterativeImputer
except:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import Iterative_Imputer

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import robust_scale, OneHotEncoder, LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline


def assemble_numeric_pipeline(variance_threshold=0.0, 
                              imputer='mean', 
                              multivariate_imputer=False, 
                              add_indicator=True,
                              quantile_transformer='normal',
                              scaler=True):
    numeric_pipeline = []
    if variance_threshold is not None:
        if isinstance(variance_threshold, float):
            numeric_pipeline.append(('var_filter', 
                                     VarianceThreshold(threshold=variance_threshold)))
        else:
            numeric_pipeline.append(('var_filter',
                                     VarianceThreshold()))
    if imputer is not None:
        if multivariate_imputer is True:
            numeric_pipeline.append(('imputer', 
                                     IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=100, n_jobs=-2), 
                                                      initial_strategy=imputer,
                                                      add_indicator=add_indicator)))
        else:
            numeric_pipeline.append(('imputer', 
                                     SimpleImputer(strategy=imputer, 
                                                   add_indicator=add_indicator)
                                    )
                                   )

    if quantile_transformer is not None:
        numeric_pipeline.append(('transformer',
                                 QuantileTransformer(n_quantiles=100, 
                                                     output_distribution=quantile_transformer, 
                                                     random_state=42)
                                )
                               )

    if scaler is not None:
        numeric_pipeline.append(('scaler', 
                                 StandardScaler()
                                )
                               )

    return Pipeline(steps=numeric_pipeline)

class ToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        return X.astype(str)
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

class DateProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, date_format='%d/%m/%Y', hours_secs=False):
        self.format = date_format
        self.columns = None
        # see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
        self.time_transformations = [
            ('day_sin', lambda x: np.sin(2*np.pi*x.dt.day/31)),
            ('day_cos', lambda x: np.cos(2*np.pi*x.dt.day/31)),
            ('dayofweek_sin', lambda x: np.sin(2*np.pi*x.dt.dayofweek/6)),
            ('dayofweek_cos', lambda x: np.cos(2*np.pi*x.dt.dayofweek/6)),
            ('month_sin', lambda x: np.sin(2*np.pi*x.dt.month/12)),
            ('month_cos', lambda x: np.cos(2*np.pi*x.dt.month/12)),
            ('year', lambda x: (x.dt.year - x.dt.year.min()) / (x.dt.year.max() - x.dt.year.min()))
        ]
        if hours_secs:
            self.time_transformations = [
                ('hour_sin', lambda x: np.sin(2*np.pi*x.dt.hour/23)),
                ('hour_cos', lambda x: np.cos(2*np.pi*x.dt.hour/23)),
                ('minute_sin', lambda x: np.sin(2*np.pi*x.dt.minute/59)),
                ('minute_cos', lambda x: np.cos(2*np.pi*x.dt.minute/59))
            ] + self.time_transformations
    
    def fit(self, X, y=None, **fit_params):
        self.columns = self.transform(X.iloc[0:1,:]).columns
        return self
    
    def transform(self, X, y=None, **fit_params):
        transformed = list()
        for col in X.columns:
            time_column = pd.to_datetime(X[col], format=self.format)
            for label, func in self.time_transformations:
                transformed.append(func(time_column))
                transformed[-1].name = transformed[-1].name + '_' + label
        transformed = pd.concat(transformed, axis=1)
        return transformed
            
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X) 

class LEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.encoders = dict()
        self.dictionary_size = list()
        self.unk = -1
    
    def fit(self, X, y=None, **fit_params):
        for col in range(X.shape[1]):
            le = LabelEncoder()
            le.fit(X.iloc[:, col].fillna('_nan'))
            le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
            
            if '_nan' not in le_dict:
                max_value = max(le_dict.values())
                le_dict['_nan'] = max_value
            
            max_value = max(le_dict.values())
            le_dict['_unk'] = max_value
            
            self.unk = max_value
            self.dictionary_size.append(len(le_dict))
            col_name = X.columns[col]
            self.encoders[col_name] = le_dict
            
        return self
    
    def transform(self, X, y=None, **fit_params):
        output = list()
        for col in range(X.shape[1]):
            col_name = X.columns[col]
            le_dict = self.encoders[col_name]
            emb = X.iloc[:, col].fillna('_nan').apply(lambda x: le_dict.get(x, le_dict['_unk'])).values
            output.append(pd.Series(emb, name=col_name).astype(np.int32))
        return output

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    
class TabularTransformer(BaseEstimator, TransformerMixin):
    
    def instantiate(self, param):
        if isinstance(param, str):
            return [param]
        elif isinstance(param, list):
            return param
        else:
            return None
    
    def __init__(self, numeric=None, dates=None, ordinal=None, cat=None, highcat=None,
                 variance_threshold=0.0, missing_imputer='mean', use_multivariate_imputer=False,
                 add_missing_indicator=True, quantile_transformer='normal', scaler=True,
                 ordinal_categories='auto', date_format='%d/%m/%Y', hours_secs=False):
        
        self.numeric = self.instantiate(numeric)
        self.dates = self.instantiate(dates)
        self.ordinal = self.instantiate(ordinal)
        self.cat  = self.instantiate(cat)
        self.highcat = self.instantiate(highcat)
        self.columns = None
        self.vocabulary = None
        
        self.numeric_process = assemble_numeric_pipeline(variance_threshold=variance_threshold, 
                                                         imputer=missing_imputer, 
                                                         multivariate_imputer=use_multivariate_imputer, 
                                                         add_indicator=add_missing_indicator,
                                                         quantile_transformer=quantile_transformer,
                                                         scaler=scaler)
        self.dates_process = DateProcessor(date_format=date_format, hours_secs=hours_secs)
        self.ordinal_process = FeatureUnion([('ordinal', OrdinalEncoder(categories=ordinal_categories)),
                                             ('categorial', Pipeline(steps=[('string_converter', ToString()),
                                                  ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]))])
        self.cat_process = Pipeline(steps=[('string_converter', ToString()),
                                           ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                           ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        self.highcat_process = LEncoder()
        
    def fit(self, X, y=None, **fit_params):
        self.columns = list()
        if self.numeric:
            self.numeric_process.fit(X[self.numeric])
            self.columns += derive_numeric_columns(X[self.numeric], 
                                                   self.numeric_process).to_list()
        if self.dates:
            self.dates_process.fit(X[self.dates])
            self.columns += self.dates_process.columns.to_list()
        if self.ordinal:
            self.ordinal_process.fit(X[self.ordinal])
            self.columns += self.ordinal + derive_ohe_columns(X[self.ordinal], 
                                                             self.ordinal_process.transformer_list[1][1])
        if self.cat:
            self.cat_process.fit(X[self.cat])
            self.columns += derive_ohe_columns(X[self.cat], 
                                               self.cat_process)
        if self.highcat:
            self.highcat_process.fit(X[self.highcat])
            self.vocabulary = dict(zip(self.highcat, self.highcat_process.dictionary_size))
            self.columns = [self.columns, self.highcat]
        return self
    
    def transform(self, X, y=None, **fit_params):
        flat_matrix = list()
        if self.numeric:
            flat_matrix.append(self.numeric_process.transform(X[self.numeric])
                               .astype(np.float32))
        if self.dates:
            flat_matrix.append(self.dates_process.transform(X[self.dates])
                               .values
                               .astype(np.float32))
        if self.ordinal:
            flat_matrix.append(self.ordinal_process.transform(X[self.ordinal])
                               .todense()
                               .astype(np.float32))
        if self.cat:
            flat_matrix.append(self.cat_process.transform(X[self.cat])
                               .todense()
                               .astype(np.float32))
        if self.highcat:
            cat_vectors = self.highcat_process.transform(X[self.highcat])
            if len(flat_matrix) > 0:
                return [np.hstack(flat_matrix)] + cat_vectors
            else:
                return cat_vectors
        else:
            return np.hstack(flat_matrix)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

