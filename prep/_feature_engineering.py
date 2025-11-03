import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Grouper(BaseEstimator, TransformerMixin):
    """
    Новые признаки на основе аггрегаций по группам.
    
    Parameters
    ----------
    group_cols : str/list
        Группирующий столбец или столбцы
        
    agg_col : str/*list
        Аггрегирующий столбец
        str if multigroup=False
        str or list if multigroup=True
        
    multigroup : bool, [default=False]
        True: Группировка по всем, переданным группирующим столбцам
        False: Группировка по одному из переданных группирующих столбцов
        
    agg_func : str, [default='mean']
        Аггрегирующая функция
         'mean'
         'count'
         'sum'
         'median'
         'max'
         'min'

    return_full : bool, [default=True]
        True: Возвращает исходный DataFrame с новыми группированными признаками
        False: Возвращает только новые признаки
    """
    def __init__(self, group_cols, agg_col, multigroup=False, agg_func='mean', return_full=True):
        self.group_cols = group_cols
        self.agg_col = agg_col
        self.multigroup = multigroup
        self.agg_func = agg_func
        self.return_full = return_full
        
    def _normalize_inputs(self):
        # make group_cols always a list of strings
        if isinstance(self.group_cols, str):
            self._group_cols = [self.group_cols]
        elif isinstance(self.group_cols, (list, tuple)):
            self._group_cols = list(self.group_cols)
        else:
            raise ValueError("group_cols must be str or list/tuple of str")

        # agg_col can be str or list
        if isinstance(self.agg_col, str):
            self._agg_cols = [self.agg_col]
        elif isinstance(self.agg_col, (list, tuple)):
            self._agg_cols = list(self.agg_col)
        else:
            raise ValueError("agg_col must be str or list/tuple of str")

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas DataFrame")
            
        self._normalize_inputs()
        X_copy = X.copy()

        if any(col not in X_copy.columns for col in self._agg_cols):
            if y is not None and len(self._agg_cols) == 1:
                # set column from y
                X_copy[self._agg_cols[0]] = y
            else:
                missing = [c for c in self._agg_cols if c not in X_copy.columns]
                raise ValueError(f"agg_col(s) not found in X: {missing}")

        # check group_cols existence
        missing_groups = [c for c in self._group_cols if c not in X_copy.columns]
        if missing_groups:
            raise ValueError(f"group_cols not found in X: {missing_groups}")

        # check NaN in group columns
        if X_copy[self._group_cols].isnull().any().any():
            raise ValueError("Found NaN in group_cols")

        # check NaN in agg columns
        for c in self._agg_cols:
            if X_copy[c].isnull().any():
                raise ValueError(f"Found NaN(s) in agg_col '{c}'")

        self.new_cols_ = []
        if self.multigroup:
            if len(self._agg_cols) == 1:
                agg_col = self._agg_cols[0]
                out_name = f"group_{'_'.join(self._group_cols)}_{str(self.agg_func).upper()}_{agg_col}"
                df_multi = (
                    X_copy.groupby(self._group_cols)[agg_col]
                    .agg(self.agg_func)
                    .rename(out_name)
                    .reset_index()
                )
                self.multi_ = df_multi
                self.new_cols_ = [out_name]
            else:
                agg_dict = {c: self.agg_func for c in self._agg_cols}
                df_multi = X_copy.groupby(self._group_cols).agg(agg_dict).reset_index()
                rename_map = {c: f"group_{'_'.join(self._group_cols)}_{str(self.agg_func).upper()}_{c}"
                              for c in self._agg_cols}
                df_multi = df_multi.rename(columns=rename_map)
                self.multi_ = df_multi
                self.new_cols_ = [rename_map[c] for c in self._agg_cols]
        else:
            if len(self._agg_cols) != 1:
                raise ValueError("When multigroup=False, agg_col must be a single column (str)")
            agg_col = self._agg_cols[0]
            self._group_maps = []
            for grp in self._group_cols:
                ser = X_copy.groupby(grp)[agg_col].agg(self.agg_func)
                new_name = f"group_{grp}_{str(self.agg_func).upper()}_{agg_col}"
                ser = ser.rename(new_name)
                self._group_maps.append((grp, ser))
                self.new_cols_.append(new_name)

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas DataFrame")
        
        X_copy = X.copy()
        
        if self.multigroup:
            out = X_copy.merge(self.multi_, on=self._group_cols, how='left')
        else:
            out = X_copy
            for grp_col, ser in self._group_maps:
                df_map = ser.reset_index()
                out = out.merge(df_map, on=grp_col, how='left')
                
        if self.return_full:
            return out
        else:
            return out[self.new_cols_]