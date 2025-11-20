import sys
import time
import functools
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import optuna
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from optuna.samplers import TPESampler

import catboost as cb
import xgboost as xgb
import lightgbm as lgbm

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

# =================================================================================================

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

    use_pruner : bool [default=True]

    pruner_name : str [default='hyperband']
        hyperband or median
        
    pruner_log : bool [default=True]
        aftermath mini log
        
    cv : bool, [default=True]
        True: using cross_validation
        False: using eval_set

    early_stopping_rounds : int, [default=100]
    
    n_folds : int, [default=5]
    
    shuffle : bool, [default=True]
    
    stratified : bool, [default=False]
    
    seed : int, [default=0]
    
    n_jobs : int, [default=-1]

    max_resource_cap: int [default=50_000]
        param for pruner
    
    """
    def __init__(self, param_space, n_trials=100, sampler=None, use_pruner=True, pruner_name='hyperband', pruner_log=True,
                 cv=True, early_stopping_rounds=100, n_folds=5, shuffle=True, stratified=False,
                 seed=0, n_jobs=-1, max_resource_cap=50000):

        self.param_space = param_space
        self.n_trials = n_trials
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.stratified = stratified
        self.seed = seed
        self.n_jobs = n_jobs
        
        self.use_pruner = use_pruner
        self.pruner_name = pruner_name
        self.pruner_log = pruner_log
        self.max_resource_cap = int(max_resource_cap)

        self.n_startup_trials = int(self.n_trials * 0.1)
        if sampler is None:
            self.sampler = TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed)
        else:
            self.sampler = sampler

    # --------------------------------------------------------------------------
    def __extract_max_resource_from_params(self, params: Dict[str, Any], default: int = 20000) -> int:
        """Try to get the max_resource (iterations / n_estimators) from params dict."""
        if not isinstance(params, dict):
            return default
        for key in ('iterations', 'n_estimators', 'num_boost_round'):
            if key in params:
                try:
                    return int(params[key])
                except Exception:
                    return default
        return default
    # -------- tqdm widget  -------------------------------------
    def __tqdm_study(self, n_trials, model, direction):
        model_name = model.__class__.__name__

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if direction == 'minimize':
                    best_score = float('inf')
                else:
                    best_score = float('-inf')

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

    # -------- parsing hyperparams space -------------------------------------------------
    def _parse_params(self, trial, params):
        parsed = {}
        for name, dist in params.items():
            if getattr(dist, '__class__', None) is not None and dist.__class__.__module__ == 'optuna.distributions':
                parsed[name] = trial._suggest(name, dist)
            else:
                parsed[name] = dist
        return parsed

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

    # -------- create pruner wrapper ---------------------------------------
    def _make_pruner(self, params):
        max_res = self.__extract_max_resource_from_params(params, default=20000)
        max_res = min(max_res, self.max_resource_cap)
        if not self.use_pruner:
            return optuna.pruners.NopPruner()

        if self.pruner_name.lower() == 'hyperband':
            return optuna.pruners.HyperbandPruner(min_resource=10, max_resource=max_res, reduction_factor=3)
        elif self.pruner_name.lower() == 'median':
            return optuna.pruners.MedianPruner(n_startup_trials=self.n_startup_trials)
        else:
            raise ValueError('if use_pruner==True pruner name should be either "hyperband" or "median" ')

    # -------- reports and prunes -------------------------
    def _report_and_maybe_prune(self, trial, values, direction='maximize'):
        """Generic: values is iterable of metric values per iteration."""
        for i, v in enumerate(values):
            trial.report(v, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # -------- CatBoost objective ---------------------------
    def _catboost_objective(self, trial, params, pool, direction):
        param_space = self._parse_params(trial, params)
        if 'iterations' not in param_space:
            param_space['iterations'] = 20000

        try:
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
        except Exception as e:
            trial.set_user_attr('error', str(e))
            raise

        metric_name = f"test-{param_space['eval_metric']}-mean"
        values = cv_results[metric_name].tolist()
        self._report_and_maybe_prune(trial, values, direction)

        if direction == 'minimize':
            best_score = min(values)
            best_iter = int(np.argmin(values))
        else:
            best_score = max(values)
            best_iter = int(np.argmax(values))

        trial.set_user_attr('best_iter', best_iter)
        return best_score

    # -------- XGBoost objective ----------------------------------------------
    def _xgboost_objective(self, trial, params, dtrain, direction):
        param_space = self._parse_params(trial, params)
        num_boost_round = int(param_space.pop('n_estimators', 1000))

        cv_results = xgb.cv(
            params=param_space,
            dtrain=dtrain,
            early_stopping_rounds=self.early_stopping_rounds,
            nfold=self.n_folds,
            shuffle=self.shuffle,
            seed=self.seed,
            stratified=self.stratified,
            num_boost_round=num_boost_round,
            as_pandas=True,
            verbose_eval=False
        )
        metric_name = [c for c in cv_results.columns if c.startswith('test') and c.endswith('mean')][0]
        values = cv_results[metric_name].tolist()
        self._report_and_maybe_prune(trial, values, direction)

        if direction == 'minimize':
            best_score = min(values)
            best_iter = int(np.argmin(values))
        else:
            best_score = max(values)
            best_iter = int(np.argmax(values))

        trial.set_user_attr('best_iter', best_iter)
        return best_score

    # -------- LightGBM objective ---------------------------------------------
    def __lightgbm_objective(self, trial, params, train_set, direction):
        param_space = self._parse_params(trial, params)
        num_boost_round = int(param_space.pop('n_estimators', 1000))

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

        metric_key = [k for k in cv_results.keys() if k.endswith('-mean')][0]
        values = list(cv_results[metric_key])
        self._report_and_maybe_prune(trial, values, direction)

        if direction == 'minimize':
            best_score = min(values)
            best_iter = int(np.argmin(values))
        else:
            best_score = max(values)
            best_iter = int(np.argmax(values))

        trial.set_user_attr('best_iter', best_iter)
        return best_score

    # -------- sklearn (no pruning) ----------------------------------
    def __else_objective(self, trial, params, model, X, y, direction):
        param_space = self._parse_params(trial, params)
        scoring = param_space.pop('scoring', None)
        if self.stratified:
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.seed)
        else:
            cv = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.seed)

        if scoring is None:
            scoring = 'roc_auc' if self.stratified else 'neg_root_mean_squared_error'

        model_ = model(**param_space)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            score = cross_val_score(model_, X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs)

        return np.abs(np.mean(score))

    # -------- Main search loop -----------------
    def search(self, X, y, catboost_kwargs=None, xgboost_kwargs=None, lightgbm_kwargs=None):
        catboost_kwargs = catboost_kwargs or {}
        xgboost_kwargs = xgboost_kwargs or {}
        lightgbm_kwargs = lightgbm_kwargs or {}

        self.best_params_ = {}
        self.best_scores_ = {}
        self.studies_ = {}

        for model, params in self.param_space.items():
            direction = self._define_direction(params)

            if 'catboost' in model().__class__.__module__:
                pool = cb.Pool(X, y, **catboost_kwargs)
                objective = functools.partial(self._catboost_objective, params=params, pool=pool, direction=direction)
                max_resource = self.__extract_max_resource_from_params(params, default=20000)
            elif 'xgboost' in model().__class__.__module__:
                dtrain = xgb.DMatrix(X, y, **xgboost_kwargs)
                objective = functools.partial(self._xgboost_objective, params=params, dtrain=dtrain, direction=direction)
                max_resource = self.__extract_max_resource_from_params(params, default=10000)
            elif 'lightgbm' in model().__class__.__module__:
                train_set = lgbm.Dataset(X, y, **lightgbm_kwargs)
                objective = functools.partial(self.__lightgbm_objective, params=params, train_set=train_set, direction=direction)
                max_resource = self.__extract_max_resource_from_params(params, default=10000)
            else:
                objective = functools.partial(self.__else_objective, params=params, model=model, X=X, y=y, direction=direction)
                max_resource = 1

            # build pruner
            pruner = self._make_pruner(params)

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            @self.__tqdm_study(n_trials=self.n_trials, model=model(), direction=direction)
            def run_optuna(tqdm_callback=None):
                study = optuna.create_study(direction=direction, sampler=self.sampler, pruner=pruner)
                study.optimize(objective, n_trials=self.n_trials, callbacks=[tqdm_callback])
                return study

            study = run_optuna()

            best_params = study.best_trial.params.copy()
            if 'catboost' in model().__class__.__module__:
                best_params['iterations'] = study.best_trial.user_attrs.get('best_iter', best_params.get('iterations', 20000))
            elif 'xgboost' in model().__class__.__module__ or 'lightgbm' in model().__class__.__module__:
                best_params['n_estimators'] = study.best_trial.user_attrs.get('best_iter', best_params.get('n_estimators', 1000))

            self.best_params_[model] = best_params
            self.best_scores_[model] = study.best_value
            self.studies_[model] = study

            if self.pruner_log:
                pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
                completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)

                print(f"[{model.__name__}]  pruned: {pruned}  | completed: {completed}")
                print('')
                print('========================================================================================================================')
                
