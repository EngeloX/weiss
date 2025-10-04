import sys
import functools
import warnings
from contextlib import contextmanager

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import BayesianRidge, LogisticRegression

import optuna
from optuna.samplers import TPESampler
from optuna.integration import OptunaSearchCV
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution, LogUniformDistribution
from optuna.pruners import MedianPruner

from joblib import Parallel, delayed

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

    parallel: bool, default = False
        Распараллеливание процессов

    n_jobs: int, default = -1
        Количество ядер процессора
    ----------------------------------------------------------------------------------------    
    Attribs
    --------
    models_ : dict
        Dictionary of trained base models (clones if clone_models=True).
    meta_model: estimator
        Trained meta-model.
    
    """
    
    
    def __init__(self, models, task='REGRESSION', kf=None, clone_models=True, parallel=False, n_jobs=-1):
        self._estimator_type = "regressor" if task=='REGRESSION' else "classifier"
        self.models = models            
        self.task = task
        self.kf = kf if kf is not None else KFold(n_splits=5, shuffle=True, random_state=42)
        self.clone_models = clone_models
        self.n_jobs = n_jobs
        self.parallel = parallel
        
        if self.task == 'REGRESSION':
            self.meta_model = BayesianRidge()
        elif self.task == 'CLASSIFICATION':
            self.meta_model = LogisticRegression(max_iter=1000)
        else:
            raise ValueError("task should be 'REGRESSION' or 'CLASSIFICATION'")
            
    @contextmanager
    def __ignore_user_warnings(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            yield    
            
    def __fit_fold(self, model, X_train, y_train, X_val):
        with self.__ignore_user_warnings():
            model.fit(X_train, y_train)
            return model.predict(X_val)
        
    def __model_predict(self, model, X):
        with self.__ignore_user_warnings():
            return model.predict(X)
        
    def __model_predict_proba(self, model, X):
        with self.__ignore_user_warnings():
            return model.predict_proba(X)[:, 1]
        
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Клонирование моделей
        if self.clone_models:
            self.models_ = {name: clone(model) for name, model in self.models.items()}
        else:
            self.models_ = self.models

        # Донастройка моделей
        for name, model in self.models_.items():
            if name == 'LGBM':
                model.set_params(verbose=-1)
            if hasattr(model, 'set_params') and "n_jobs" in model.get_params():
                model.set_params(n_jobs=self.n_jobs)
    
        # Массивы для хранения предсказаний базовых моделей
        all_base_models_preds = np.zeros((X.shape[0], len(self.models_)))
    
        for i, (name, model) in enumerate(self.models_.items()):
            if self.parallel:
                # сохраняем сплиты
                splits = list(self.kf.split(X))
                results = Parallel(n_jobs=self.n_jobs, backend='threading')(
                    delayed(self.__fit_fold)(clone(model), X[train_idx], y[train_idx], X[val_idx])
                    for train_idx, val_idx in splits
                )
                for (train_idx, val_idx), fold_pred in zip(splits, results):
                    all_base_models_preds[val_idx, i] = fold_pred
            else:
                model_preds = np.zeros( X.shape[0] )          
                for train_idx, val_idx in self.kf.split(X):
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]
                    
                    model_preds[val_idx] = self.__fit_fold(model, X_train, y_train, X_val)
                # Сохранение результатов для метомодели
                all_base_models_preds[:, i] = model_preds
    
            # дообучение модели на всей выборке
            model.fit(X, y)
    
        # обучение мета-модели
        self.meta_model.fit(all_base_models_preds, y)
    
        return self

    
    def predict(self, X):
        X = np.asarray(X) 
        
        if self.task == 'REGRESSION':
            if self.parallel:
                all_base_models_preds = np.column_stack(
                Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self.__model_predict)(model, X) for model in self.models_.values()))
            else:
                all_base_models_preds = np.zeros( (X.shape[0], len(self.models_)) )
                for i, (name, model) in enumerate(self.models_.items()):
                    model_preds = self.__model_predict(model, X)
                    all_base_models_preds[:, i] = model_preds 
                    
            ensemble_preds = self.meta_model.predict(all_base_models_preds)
            
        elif self.task == 'CLASSIFICATION':
            probs = self.predict_proba(X)
            return (probs >= 0.5).astype(int)
            
        return ensemble_preds    
        
    def predict_proba(self, X):
        X = np.asarray(X)

        # Если пытаться параллелить
        if self.parallel:
            all_base_models_preds = np.column_stack(
                Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self.__model_predict_proba)(model, X) for model in self.models_.values())
            )
        # Без паралеливания
        else:
            all_base_models_preds = np.zeros( (X.shape[0], len(self.models_)) )
            for i, (name, model) in enumerate(self.models_.items()):
                model_probas = self.__model_predict_proba(model, X)
                all_base_models_preds[:, i] = model_probas
                
        ensemble_preds = self.meta_model.predict_proba(all_base_models_preds)[:, 1]
        
        return ensemble_preds
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

class old_Stacker(BaseEstimator):
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
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
class Searcher:
    """
    """
    def __init__(self, models, param_spaces, optimizer='optuna', n_trials=10, timeout=600, random_state=100):
        self.models = models
        self.param_spaces = param_spaces
        self.optimizer = optimizer
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        
        if self.param_spaces.keys() != self.models.keys():
            raise ValueError('Ключи(названия моделей) в models не соответствуют ключам(названиям моделей) в param_spaces')
    # ------------------------------------    
    @contextmanager
    def __ignore_user_warnings(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            yield  
    # --------------------------------------        
    def search(self, X_train, y_train, X_valid=None, y_valid=None, optuna_sampler=None, optuna_pruner=None):
        
        # Если валидационаня выборка не будет передана, она создастся
        if (X_train is None) or (y_valid is None):
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, random_state=self.random_state)

        # Клонирование моделей чтобы не загрязнять переданные
        self.models_ = {name: clone(model) for name, model in self.models.items()} 
        self.best_params_ = {} # Пустой словарь для хранения лучших найденных параметров
        self.best_score_ = {} # Пустой словарь для хранения лучших результатов

        # Небольшая донастройка моделей, чтобы не мусорить выводом
        for name, model in self.models_.items():
            if name == 'LGBM':
                model.set_params(verbose=-1)
            if name == 'XGB':
                model.set_params(verbosity=0)
                
        # Для оптюны
        if self.optimizer == 'optuna':
            if optuna_sampler is None:
                self.optuna_sampler = TPESampler(seed=self.random_state)
            if optuna_pruner is None:
                self.optuna_pruner=MedianPruner(n_warmup_steps=5)
                
            self.best_params_, self.best_score_ = self.__optuna_search(X_train, y_train, X_valid, y_valid, 
                                                                       sampler=self.optuna_sampler, pruner=self.optuna_pruner)
        # для hyperopt
        if self.optimizer == 'hyperopt':
            pass
            
        
    # -------------------------------------------------------------
    def __hyperopt_search():
        pass
    # -------------------------------------------------------------    
    def __optuna_search(self, X_train, y_train, X_valid, y_valid, sampler, pruner):
        optuna.logging.disable_default_handler()
        self.sampler = sampler
        self.pruner = pruner
        i = 0 
        for name in self.models_:
            param_space = self.param_spaces[name]
            # Objective функция
            def objective(trial):
                params = {}
                for param_name, param_range in param_space.items():
                    params[param_name] = trial._suggest(param_name, param_range)
                self.models_[name].set_params(**params)
                # ------------- обучение моделей -------------
                if name == 'XGB':
                    self.models_[name].fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                elif name == 'LGBM':
                    self.models_[name].fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
                else:
                    self.models_[name].fit(X_train, y_train)
                # -------------------------------------------
                # Костыль от бага "UserWarning: X does not have valid feature names...""
                with self.__ignore_user_warnings(): 
                    score = self.models_[name].score(X_valid, y_valid)

                return score
            study = optuna.create_study(sampler=self.sampler, direction='maximize', pruner=self.pruner)
            @self.__tqdm_study(n_trials=self.n_trials)
            def run_study(study, objective, n_trials, timeout, callbacks):
                study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks)
            desc = f"[{name}] Model {i+1}/{len(self.models_)}"
            run_study(study, objective, n_trials=self.n_trials, timeout=self.timeout, desc=desc)
            best = study.best_trial
            self.best_params_[name] = best.params
            self.best_score_[name] = best.value
            self.models_[name].set_params(**best.params)
            i+=1
        return self.best_params_, self.best_score_
    # ---------------------------------------------------------------------------------------------------
    def __tqdm_study(self, n_trials):
        """
        Декоратор для оборачивания функции, которая вызывает study.optimize,
        и добавления прогресс-бара tqdm.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, desc=None, **kwargs):
                # tqdm-progress bar
                with tqdm(total=n_trials, desc=desc if desc else "Optimization",
                          file=sys.stdout, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as progress_bar:
                    def tqdm_callback(study, trial):
                        progress_bar.update(1)
                        progress_bar.set_postfix({"best_score": study.best_value})
                    if "callbacks" in kwargs and kwargs["callbacks"] is not None:
                        kwargs["callbacks"].append(tqdm_callback)
                    else:
                        kwargs["callbacks"] = [tqdm_callback]
                    return func(*args, **kwargs)
            return wrapper
        return decorator