import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from category_encoders import OneHotEncoder

import catboost as cb
import xgboost as xgb
import lightgbm as lgbm

import optuna
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from optuna.samplers import TPESampler, GridSampler
from optuna.pruners import HyperbandPruner, MedianPruner

import functools
import time
from tqdm import tqdm
import sys
import warnings; warnings.filterwarnings('ignore')

class Searcher():
    """
    Parameters
    ----------
    param_space : dict
        keys: model class(not object) | values: params for searching
        example: {xgb.XGBRegressor: {'eta': FloatDistribution(0.001, 0.1, log=True)},
                  cb.CatBoostRegressor: {'depth': IntDistribution(2, 8)}}

    n_trials : int [default=100]
        The number of trials for each process.
        
    sampler : optuna.sampler [default=TPESampler()]
        n_startup_trials for default TPESampler() is 10% of n_trials
        
    pruner : optuna.pruner [default=None]

    cv : bool
        True: using cross_validation
        False: using eval_set

    early_stopping_rounds: int [default=100]
    
    nfold: int [default=5]
    
    shuffle: bool [default=True]
    
    random_state: int
    """
    def __init__(self, param_space, n_trials=100, sampler=None, pruner=None,
                 cv=True, early_stopping_rounds=100, nfold=5, shuffle=True, random_state=0):
        self.param_space = param_space
        self.n_trials = n_trials
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.nfold = nfold
        self.shuffle = shuffle
        self.random_state = random_state

        if sampler is None:
            self.sampler = TPESampler(n_startup_trials=int(self.n_trials*0.1),
                                      seed=self.random_state)
        else:
            self.sampler = sampler
            
        self.pruner = pruner
        # ----
        self.__classification_loss = ['logloss']
        self.__regression_loss = ['rmse']
        
        self.__minimize_metrics = ['rmse']
        self.__maximize_metrics = ['auc']
        
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
                        pbar.set_postfix_str(f"time: {time_str} | best_score: {best_score}")
                        pbar.update(1)
    
                    return func(tqdm_callback=tqdm_callback, *args, **kwargs)
            return wrapper
        return decorator
    # ---------------------------------------    
    def __catboost_cv(self, param_space, pool, early_stopping_rounds, fold_count, shuffle, random_state, direction):
        if param_space['objective'] in self.__classification_loss:
            stratified = True
        else:
            stratified = False
            
        cv_results = cb.cv(
            params = param_space,
            pool = pool,
            early_stopping_rounds = early_stopping_rounds,
            fold_count = fold_count,
            shuffle = shuffle,
            partition_random_seed = random_state,
            stratified = stratified,
            logging_level = 'Silent'
        )
        if direction == 'minimize':
            best_score = cv_results[f"test-{param_space['eval_metric']}-mean"].min()
            best_iter = cv_results['iterations'].max()
        else:
            best_score = cv_results[f"test-{param_space['eval_metric']}-mean"].max()
            best_iter = cv_results['iterations'].min()

        return best_score, best_iter
    # ---------------------------------------
    def __xgboost_cv(self, param_space, dtrain, early_stopping_rounds, nfold, shuffle, seed, direction):
        if param_space['objective'] in self.__classification_loss:
            stratified = True
        else:
            stratified = False
            
        # Разные варики итераций
        if 'n_estimators' in param_space:
            num_boost_round = param_space['n_estimators']
        elif 'iterations' in param_space:
            num_boost_round = param_space['iterations']
        else:
            num_boost_round = 1_000
            
        cv_results = xgb.cv(
            params = param_space,
            dtrain = dtrain,
            early_stopping_rounds = early_stopping_rounds,
            nfold = nfold,
            shuffle = shuffle,
            seed = seed,
            stratified = stratified,
            num_boost_round = num_boost_round
        )
        if direction == 'minimize':
            best_score = cv_results[f"test-{param_space['eval_metric']}-mean"].min()
            best_iter = cv_results[f"test-{param_space['eval_metric']}-mean"].idxmin() + 1
        else:
            best_score = cv_results[f"test-{param_space['eval_metric']}-mean"].max()
            best_iter = cv_results[f"test-{param_space['eval_metric']}-mean"].idxmax() + 1

        return best_score, best_iter
    # -------------------------------------------
    def __lgbm_cv(self, param_space, train_set, early_stopping_rounds, nfold, shuffle, seed, direction):
        if param_space['objective'] in self.__classification_loss:
            stratified = True
        else:
            stratified = False

        # Разные варики итераций
        if 'n_estimators' in param_space:
            num_boost_round = param_space['n_estimators']
        elif 'iterations' in param_space:
            num_boost_round = param_space['iterations']
        else:
            num_boost_round = 1_000
            
        cv_results = lgbm.cv(
            params = param_space,
            train_set = train_set,
            nfold = nfold,
            shuffle = shuffle,
            stratified = stratified,
            seed = seed,
            num_boost_round = num_boost_round,
            callbacks=[lgbm.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )
        
        if direction == 'minimize':
            best_score = np.min(cv_results[f"valid {param_space['metric']}-mean"])
        else:
            best_score = np.max(cv_results[f"valid {param_space['metric']}-mean"])
            
        best_iter = cv_results[f"valid {param_space['metric']}-mean"].index(best_score)
        return best_score, best_iter
    # -------------------------------------------  
    def search(self, X, y, catboost_kwargs=None, xgb_kwargs=None, lgbm_kwargs=None):
        catboost_kwargs = catboost_kwargs or {}
        xgb_kwargs = xgb_kwargs or {}
        lgbm_kwargs = lgbm_kwargs or {}
        
        self.best_params_ = {}
        self.best_scores_ = {}
        for model, params in self.param_space.items():
            #  define direction
            if 'eval_metric' in params:
                if params['eval_metric'].lower() in self.__minimize_metrics:
                    direction = 'minimize'
                elif params['eval_metric'].lower() in self.__maximize_metrics:
                    direction = 'maximize'
                else:
                    raise ValueError('Wrong metric for eval')
            elif 'metric' in params:
                if params['metric'].lower() in self.__minimize_metrics:
                    direction = 'minimize'
                elif params['metric'].lower() in self.__maximize_metrics:
                    direction = 'maximize'
                else:
                    raise ValueError('Wrong metric for eval')
                
            def objective(trial):
                # param sampling
                param_space = {}
                for param_name, param_dist in params.items():
                    if param_dist.__class__.__module__ == 'optuna.distributions':
                        param_space[param_name] = trial._suggest(param_name, param_dist)
                    else:
                        param_space[param_name] = trial.suggest_categorical(param_name, [param_dist])
                
  
                # Если кросс-валидация        
                if self.cv:
                    if 'catboost' in model().__class__.__module__:
                        pool = cb.Pool(X, y, **catboost_kwargs)
                        best_score, best_iter = self.__catboost_cv(param_space, pool, 
                                                                   self.early_stopping_rounds, self.nfold, self.shuffle, self.random_state,
                                                                   direction=direction) 
                        trial.set_user_attr('best_iter', best_iter)
                        return best_score
                    elif 'xgboost' in model().__class__.__module__:
                        dtrain = xgb.DMatrix(X, y, **xgb_kwargs)
                        best_score, best_iter = self.__xgboost_cv(param_space, dtrain, 
                                                                  self.early_stopping_rounds, self.nfold, self.shuffle, self.random_state, 
                                                                  direction)
                        trial.set_user_attr('best_iter', best_iter)
                        return best_score
                    elif 'lightgbm' in model().__class__.__module__:
                        train_set = lgbm.Dataset(X, y, **lgbm_kwargs)
                        best_score, best_iter = self.__lgbm_cv(param_space, train_set, 
                                                               self.early_stopping_rounds, self.nfold, self.shuffle, self.random_state, 
                                                               direction)
                        trial.set_user_attr('best_iter', best_iter)
                        return best_score
                    else:
                        pass
                # если eval_set
                else:
                    if 'catboost' in model().__class__.__module__:
                        pass
                    elif 'xgboost' in model().__class__.__module__:
                        pass
                    elif 'lightgbm' in model().__class__.__module__:
                        pass
                    else:
                        pass
            # --------- Сам поиск -----------------------------
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            @self.__tqdm_study(n_trials=self.n_trials, model=model(), direction=direction)
            def run_optuna(tqdm_callback=None):
                study = optuna.create_study(direction=direction,
                                             sampler=self.sampler)
                study.optimize(func=objective, n_trials=self.n_trials, callbacks=[tqdm_callback])
                return study
                
            study = run_optuna()
 
            best_params = study.best_params
            if 'catboost' in model().__class__.__module__:
                best_params['iterations'] = study.best_trial.user_attrs['best_iter']
            elif 'xgboost' in model().__class__.__module__ or 'lightgbm' in model().__class__.__module__:
                best_params['n_estimators'] = study.best_trial.user_attrs['best_iter']
                
            self.best_params_[model] = best_params
            self.best_scores_[model] = study.best_value