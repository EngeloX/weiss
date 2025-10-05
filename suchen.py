import warnings
from contextlib import contextmanager
import functools
import sys

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

from hyperopt import hp, fmin, tpe, Trials, space_eval
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.distributions import IntDistribution, FloatDistribution, LogUniformDistribution, CategoricalDistribution

from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------------------------------------

class IntDistrib:
    def __init__(self, low, high, q=1):
        self.low = low
        self.high = high
        self.q = q
        
class FloatDistrib:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        
class LogDistrib:
    def __init__(self, low, high):
        self.low = low
        self.high = high

class CatDistrib:
    def __init__(self, choices):
        self.choices = choices

# ----------------------------------------------------------------------------------------------------------------------------------------------------

def sample_params(optimizer, param_spaces):
    new_param_spaces = {}   
    for estimator_name, params in param_spaces.items():
        new_param_spaces[estimator_name] = {}
        for param_name, param_range in params.items():
            
            if optimizer == 'optuna':
                if isinstance(param_range, IntDistrib):
                    new_param_spaces[estimator_name][param_name] = IntDistribution(param_range.low, param_range.high)
                elif isinstance(param_range, FloatDistrib):
                    new_param_spaces[estimator_name][param_name] = FloatDistribution(param_range.low, param_range.high)
                elif isinstance(param_range, LogDistrib):
                    new_param_spaces[estimator_name][param_name] = LogUniformDistribution(param_range.low, param_range.high)
                elif isinstance(param_range, CatDistrib):
                    new_param_spaces[estimator_name][param_name] = CategoricalDistribution(param_range.choices) 
                    
            elif optimizer == 'hyperopt':
                if isinstance(param_range, IntDistrib):
                    new_param_spaces[estimator_name][param_name] = hp.quniform(param_name, param_range.low, param_range.high, q=param_range.q)
                elif isinstance(param_range, FloatDistrib):    
                    new_param_spaces[estimator_name][param_name] = hp.uniform(param_name, param_range.low, param_range.high)
                elif isinstance(param_range, LogDistrib):
                    new_param_spaces[estimator_name][param_name] = hp.loguniform(param_name, np.log(param_range.low), np.log(param_range.high))
                elif isinstance(param_range, CatDistrib):
                    new_param_spaces[estimator_name][param_name] = hp.choice(param_name, param_range.choices)
                
    return new_param_spaces

# ----------------------------------------------------------------------------------------------------------------------------------------------------

class Searcher:
    """
    Parameters
    ----------
    models: dict
        dictionary {'name': estimator }
    
    param_spaces: dict
        dictionary {'estimator_name': {'param_name': param_range, }}

    task: str, default = 'regression'
        'regression' or 'classification'
        
    optimizer: str, default = 'optuna'
        'optuna' or 'hyperopt'
        
    n_trials: int, default = 10
        Количество итераций поиска гиперпараметров
        
    metrics: callable
        default for task regression: r2_score
        default for task classification: accuracy_score
        
    early_stopping_rounds: int, default = 100
        Количество раундов ранней остановки
    
    timeout: int, default = 600
        Предел времени выполнения одного trial
        
    random_state: int, default = 100
        Состояние рандома, для воспроизводимости
    ----------------------------------------------------------------------------------------
    Attributes:
    -----------
    best_params_:
        Лучшие параметры
        
    best_score_:
        Лучшие результаты
        
    models_:
        Модели с утсановленными лучшими параметрами
        
    pruner:
        Pruner для optuna
        
    sampler:
        Sampler для optuna
    """
    def __init__(self, models, param_spaces, task='regression', optimizer='optuna', n_trials=10, metric=None, 
                 early_stopping_rounds=100, timeout=600, random_state=100):
        self.models = models
        self.param_spaces = param_spaces
        self.optimizer = optimizer.lower()
        if self.optimizer not in ['optuna', 'hyperopt']:
            raise ValueError("task should be in ['regression', 'classification'] ")
            
        self.n_trials = n_trials
        self.task = task.lower()
        if self.task not in ['regression', 'classification']:
            raise ValueError("task should be in ['regression', 'classification'] ")

        self.metric = metric
        if self.metric is None:
            if self.task == 'regression':
                self.metric = r2_score
            elif self.task == 'classification':
                self.metric = accuracy_score
            
        self.early_stopping_rounds = early_stopping_rounds
        self.timeout = timeout
        self.random_state = random_state
        
        self.param_spaces_ = sample_params(self.optimizer, self.param_spaces)
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
        if (X_valid is None) or (y_valid is None):
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
                self.sampler = TPESampler(seed=self.random_state)
            if optuna_pruner is None:
                self.pruner=MedianPruner(n_warmup_steps=5)
                
            self.__optuna_search(X_train, y_train, X_valid, y_valid, sampler=self.sampler, pruner=self.pruner)
        # для hyperopt
        elif self.optimizer == 'hyperopt':
            self.__hyperopt_search(X_train, y_train, X_valid, y_valid)
        else:
            raise ValueError('optimizer should be optuna or hyperopt')      
        
    # -------------------------------------------------------------
    def __hyperopt_search(self, X_train, y_train, X_valid, y_valid):
        for name in self.models_:
            trials = Trials()
            param_space = self.param_spaces_[name]    
            def objective(params):
                self.models_[name].set_params(**params)
                if name in ['XGB', 'LGBM']:
                    self.models_[name].set_params(early_stopping_rounds=self.early_stopping_rounds)
                if name == 'XGB':
                    self.models_[name].fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                elif name == 'LGBM':
                    self.models_[name].fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
                else:
                    self.models_[name].fit(X_train, y_train)
                # Костыль от бага "UserWarning: X does not have valid feature names...""
                with self.__ignore_user_warnings():
                    y_pred = self.models_[name].predict(X_valid)
                    if self.metric is not None:
                        score = -self.metric(y_valid, y_pred)
                    else:
                        if self.task == 'regression':
                            score = -r2_score(y_valid, y_pred)
                        else:
                            score = -accuracy_score(y_valid, y_pred)
                return score
                
            best = fmin(
                fn = objective,
                space = param_space,
                algo = tpe.suggest,
                max_evals = self.n_trials,
                rstate = np.random.default_rng(self.random_state),
                trials = trials
            )
            # Сохраняем и применяем лучшие параметры
            self.best_params_[name] = space_eval(param_space, best)
            # Кастим целые параметры
            self.models_[name].set_params(**self.best_params_[name])
            self.best_score_[name] = -trials.best_trial['result']['loss']
    # -------------------------------------------------------------    
    def __optuna_search(self, X_train, y_train, X_valid, y_valid, sampler, pruner):
        optuna.logging.disable_default_handler()
        i = 0 
        for name in self.models_:
            param_space = self.param_spaces_[name]
            # Objective функция
            def objective(trial):
                params = {}
                for param_name, param_range in param_space.items():
                    params[param_name] = trial._suggest(param_name, param_range)
                self.models_[name].set_params(**params)
                # ------------- обучение моделей -------------
                if name in ['XGB', 'LGBM']:
                    self.models_[name].set_params(early_stopping_rounds = self.early_stopping_rounds)
                    
                if name == 'XGB':
                    self.models_[name].fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                elif name == 'LGBM':
                    self.models_[name].fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
                else:
                    self.models_[name].fit(X_train, y_train)
                # -------------------------------------------
                # Костыль от бага "UserWarning: X does not have valid feature names...""
                with self.__ignore_user_warnings():
                    y_pred = self.models_[name].predict(X_valid)
                    if self.metric is not None:
                        score = self.metric(y_valid, y_pred)
                    else:
                        if self.task == 'regression':                            
                            score = r2_score(y_valid, y_pred)
                        else:                
                            score = accuracy_score(y_valid, y_pred)
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
            if name in ['XGB', 'LGBM']:
                    self.models_[name].set_params(early_stopping_rounds = None)
            i+=1
        
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
    
    