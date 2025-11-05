import sys
import time
import functools

import numpy as np
import pandas as pd

import catboost as cb
import xgboost as xgb
import lightgbm as lgbm

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.exceptions import ConvergenceWarning

from tqdm import tqdm

import optuna
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from optuna.samplers import TPESampler
import warnings

# -----

class Searcher:
    """
    Optuna hyperparameters searcher

    Parameters
    ----------
    param_space : dict
        keys: model class(not object) | values: params for searching
        example: {xgb.XGBRegressor: {'eta': FloatDistribution(0.001, 0.1, log=True)}, 
                  cb.CatBoostRegressor: {'depth': IntDistribution(2, 8)}}
    
    n_trials : int, [default=100]
        The number of trials for each process.
        
    sampler : optuna.sampler [default=TPESampler()]
        n_startup_trials for default TPESampler() is 10% of n_trials
    
    cv : bool, [default=True]
        True: using cross_validation
        False: using eval_set

    early_stopping_rounds : int, [default=100]
    
    n_folds : int, [default=5]
    
    shuffle : bool, [default=True]
    
    stratified : bool, [default=False]
    
    seed : int, [default=0]
    
    n_jobs : int, [default=-1]
        
    """
    def __init__(self, param_space, 
                 n_trials=100, sampler=None,
                 cv=True, early_stopping_rounds=100, n_folds=5, shuffle=True, stratified=False,
                 seed=0, n_jobs=-1):
        
        self.param_space = param_space
        self.n_trials = n_trials
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.stratified = stratified
        self.seed = seed
        self.n_jobs = n_jobs

        self.n_startup_trials = int(self.n_trials*0.1)
        if sampler is None:
            self.sampler = TPESampler(n_startup_trials=self.n_startup_trials,
                                      seed=self.seed)
        else:
            self.sampler = sampler
    # =====================================================================================
    def __tqdm_study(self, n_trials, model, direction): 
        model_name = model.__class__.__name__
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if direction == 'minimize':
                    best_score = float("inf")
                else:
                    best_score = float("-inf")
                    
                start_time = time.time()
                with tqdm(total=n_trials, desc=f"[{model_name}]", file=sys.stdout,
                          bar_format="{desc} {bar} {n_fmt}/{total_fmt} {postfix}") as pbar:
                    def tqdm_callback(study, trial):
                        nonlocal best_score

                        if direction == 'minimize':
                            if study.best_value < best_score:
                                best_score = study.best_value
                        else:
                            if study.best_value > best_score:
                                best_score = study.best_value
                            
                        elapsed = int(time.time() - start_time)
                        h, rem = divmod(elapsed, 3600)
                        m, s = divmod(rem, 60)
                        time_str = f"{h:02d}:{m:02d}:{s:02d}"
                        pbar.set_postfix_str(f"time: {time_str} | best_score: {np.abs(best_score)}")
                        pbar.update(1)
    
                    return func(tqdm_callback=tqdm_callback, *args, **kwargs)
            return wrapper
        return decorator
    # =====================================================================================
    @staticmethod
    def _parse_params(trial, params):
        param_space = {}
        for param_name, param_dist in params.items():
            if param_dist.__class__.__module__ == 'optuna.distributions':
                param_space[param_name] = trial._suggest(param_name, param_dist)
            else:
                param_space[param_name] = trial.suggest_categorical(param_name, [param_dist])
        return param_space

    @staticmethod
    def _define_direction(params):
        metric_key = None
        for key in ('eval_metric', 'metric', 'scoring'):
            if key in params:
                metric_key = key
                break
    
        if metric_key is None:
            return 'minimize'
    
        metric = str(params[metric_key]).lower()

        minimize_metrics = ['rmse', 'mae', 'poisson', 'logloss']
        maximize_metrics = ['r2', 'auc', 'accuracy', 'precision', 'recall', 'mape']
    
        if metric in minimize_metrics:
            return 'minimize'
        elif metric in maximize_metrics:
            return 'maximize'
        else:
            return 'minimize'
    # =====================================================================================            
    def _catboost_objective(self, trial, params, pool, direction):
        param_space = self._parse_params(trial, params)
        
        cv_results = cb.cv(
            params=param_space,
            pool=pool,
            early_stopping_rounds=self.early_stopping_rounds,
            fold_count=self.n_folds,
            shuffle=self.shuffle,
            partition_random_seed=self.seed,
            stratified=self.stratified,
            logging_level='Silent',
            metric_period=1
        )

        metric_name = f"test-{param_space['eval_metric']}-mean"
        if direction == 'minimize':
            best_score = cv_results[metric_name].min()
            best_iter = cv_results.loc[cv_results[metric_name].idxmin(), 'iterations']
        else:
            best_score = cv_results[metric_name].max()
            best_iter = cv_results.loc[cv_results[metric_name].idxmax(), 'iterations']

        trial.set_user_attr('best_iter', int(best_iter))
        return best_score
    # ===================================================================================== 
    def _xgboost_objective(self, trial, params, dtrain, direction):
        param_space = self._parse_params(trial, params)
        num_boost_round = param_space.pop('n_estimators', 1000)
        cv_results = xgb.cv(
            params=param_space,
            dtrain=dtrain,
            early_stopping_rounds=self.early_stopping_rounds,
            nfold=self.n_folds,
            shuffle=self.shuffle,
            seed=self.seed,
            stratified=self.stratified,
            num_boost_round=num_boost_round
        )
        metric_name = f"test-{param_space['eval_metric']}-mean"
        if direction == 'minimize':
            best_score = cv_results[metric_name].min()
            best_iter = int(cv_results[metric_name].idxmin())
        else:
            best_score = cv_results[metric_name].max()
            best_iter = int(cv_results[metric_name].idxmax())
        trial.set_user_attr('best_iter', int(best_iter))
        return best_score
    # =====================================================================================     
    def __lightgbm_objective(self, trial, params, train_set, direction):
        param_space = self._parse_params(trial, params)
        num_boost_round = param_space.pop('n_estimators', 1000)
        
        cv_results = lgbm.cv(
            params=param_space,
            train_set=train_set,
            nfold=self.n_folds,
            shuffle=self.shuffle,
            seed=self.seed,
            stratified=self.stratified,
            num_boost_round=num_boost_round,
            callbacks=[lgbm.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]
        )
        metric_name = f"valid {param_space['metric']}-mean"
        if direction == 'minimize':
            best_score = np.min(cv_results[metric_name])
        else:
            best_score = np.max(cv_results[metric_name])
            
        best_iter = cv_results[metric_name].index(best_score)
        trial.set_user_attr('best_iter', int(best_iter))
        return best_score
    # =====================================================================================        
    def __else_objective(self, trial, params, model, X, y, direction):
        param_space = self._parse_params(trial, params)
        scoring = param_space.pop('scoring', None)
        if self.stratified == True:
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.seed)
        else:
            cv = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.seed)
            
        if scoring is None:
            scoring = 'roc_auc' if self.stratified else 'neg_root_mean_squared_error'
            
        model_ = model(**param_space)
        
        with warnings.catch_warnings(): # Глушилка сходимости
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            score = cross_val_score(model_, X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs)

        return np.abs(np.mean(score))
    # =====================================================================================           
    def search(self, X, y, catboost_kwargs=None, xgboost_kwargs=None, lightgbm_kwargs=None):
        catboost_kwargs = catboost_kwargs or {}
        xgboost_kwargs = xgboost_kwargs or {}
        lightgbm_kwargs = lightgbm_kwargs or {}
        
        self.best_params_ = {}
        self.best_scores_ = {}
        
        
        for model, params in self.param_space.items():
            direction = self._define_direction(params)

            # === CatBoost ===
            if 'catboost' in model().__class__.__module__:
                pool = cb.Pool(X, y, **catboost_kwargs)
                objective = functools.partial(
                    self._catboost_objective,
                    params=params,
                    pool=pool,
                    direction=direction
                )
            # === XGBoost ===    
            elif 'xgboost' in model().__class__.__module__:
                dtrain = xgb.DMatrix(X, y, **xgboost_kwargs)
                objective = functools.partial(
                    self._xgboost_objective,
                    params=params,
                    dtrain=dtrain,
                    direction=direction
                )
            elif 'lightgbm' in model().__class__.__module__:
                train_set = lgbm.Dataset(X, y, **lightgbm_kwargs)
                objective = functools.partial(
                    self.__lightgbm_objective,
                    params=params,
                    train_set=train_set,
                    direction=direction
                )
            else:
                objective = functools.partial(
                    self.__else_objective,
                    params=params, 
                    model=model, 
                    X=X, 
                    y=y, 
                    direction=direction
                )
            # ======= Оптимизация ==========================================================
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            @self.__tqdm_study(n_trials=self.n_trials, model=model(), direction=direction)
            def run_optuna(tqdm_callback=None):
                study = optuna.create_study(direction=direction, sampler=self.sampler)
                study.optimize(objective, n_trials=self.n_trials, callbacks=[tqdm_callback])
                return study

            study = run_optuna()
            # == Вывод =====================================================================
            best_params = study.best_trial.params.copy()
            if 'catboost' in model().__class__.__module__:
                best_params['iterations'] = study.best_trial.user_attrs['best_iter']
            elif 'xgboost' in model().__class__.__module__ or 'lightgbm' in model().__class__.__module__:
                best_params['n_estimators'] = study.best_trial.user_attrs['best_iter']
            else:
                pass
            self.best_params_[model] = best_params
            self.best_scores_[model] = study.best_value
            