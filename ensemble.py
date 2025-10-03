import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

import warnings
from contextlib import contextmanager
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.linear_model import BayesianRidge, LogisticRegression


class Stacker(BaseEstimator):
    """
    Params:
    -------
    models: dict
        dictionary {'name': estimator }
        name for models should be ['LGBM', 'XGB']
        
    task: str, default = 'regression'
        'REGRESSION' or binary 'CLASSIFICATION'
        if 'regression': BayesianRidge()
        if 'classification': LogisticRegression()
        
    kf: sklearn.KFold
    
    clone_models: bool, default = True
        Если True, создаются копии моделей (исходные не изменяются).
        Если False, обучение проводится прямо на переданных объектах в исходном словаре.
    ----------------------------------------------------------------------------------------    
    Attribs
    --------
    models_ : dict
        Dictionary of trained base models (clones if clone_models=True).
    meta_model_ : estimator
        Trained meta-model.
    
    """
    
    
    def __init__(self, models, task='REGRESSION', kf=None, clone_models=True):
        self._estimator_type = "regressor" if task=='REGRESSION' else "classifier"
        self.models = models            
        self.task = task
        self.kf = kf if kf is not None else KFold(n_splits=5, shuffle=True, random_state=42)
        self.clone_models = clone_models
        
        if self.task == 'REGRESSION':
            self.meta_model = BayesianRidge()
        elif self.task == 'CLASSIFICATION':
            self.meta_model = LogisticRegression()
        else:
            raise ValueError("task should be 'REGRESSION' or 'CLASSIFICATION'")
            
    @contextmanager
    def __ignore_user_warnings(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            yield    
            
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Клонирование моделей
        if self.clone_models:
            self.models_ = {name: clone(model) for name, model in self.models.items()}
        else:
            self.models_ = self.models

        # Переключение Verbose
        if 'LGBM' in self.models_:
            self.models_['LGBM'].set_params(verbose=-1)
        
        # Массивы для хранения предсказаний базовых моделей
        all_base_models_preds = np.zeros( (X.shape[0], len(self.models_)) )
        
        # Обучение базовых моделей на KFold      
        for i, (name, model) in enumerate(self.models_.items()):
            model_preds = np.zeros( X.shape[0] )          
            for train_idx, val_idx in self.kf.split(X):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                # Перехват варнинга из за бага "UserWarning: X does not have valid feature names...""
                with self.__ignore_user_warnings():
                    model.fit(X_train, y_train)
                    model_preds[val_idx] = model.predict(X_val)
            # Сохранение результатов для метомодели
            all_base_models_preds[:, i] = model_preds
            # Дообучение моделей на всей выборке X
            model.fit(X, y)
        # Мета-модель    
        self.meta_model.fit(all_base_models_preds, y)
        return self
    
    def predict(self, X):
        X = np.asarray(X) 
        all_base_models_preds = np.zeros( (X.shape[0], len(self.models_)) )
        if self.task == 'REGRESSION':
            for i, (name, model) in enumerate(self.models_.items()):
                # Перехват варнинга из за бага "UserWarning: X does not have valid feature names...""
                with self.__ignore_user_warnings():
                    model_preds = model.predict(X)
                all_base_models_preds[:, i] = model_preds
            ensemble_preds = self.meta_model.predict(all_base_models_preds)
            
        elif self.task == 'CLASSIFICATION':
            probs = self.predict_proba(X)
            return (probs >= 0.5).astype(int)
            
        return ensemble_preds    
        
    def predict_proba(self, X):
        X = np.asarray(X)
        all_base_models_preds = np.zeros( (X.shape[0], len(self.models_)) )
        for i, (name, model) in enumerate(self.models_.items()):
            # Перехват варнинга из за бага "UserWarning: X does not have valid feature names...""
            with self.__ignore_user_warnings():
                model_probas = model.predict_proba(X)[:, 1]
            all_base_models_preds[:, i] = model_probas
        ensemble_preds = self.meta_model.predict_proba(all_base_models_preds)[:, 1]
        
        return ensemble_preds
    
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------