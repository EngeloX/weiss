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

# ==================================================================================================================================================================

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class PCACreator(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, num_features=None, optimum=0.9, return_df=False):
        self.n_components = n_components
        self.num_features = num_features
        self.optimum = optimum
        self.return_df = return_df
        
    def fit(self, X, y=None):
        X_standarded = X.copy()
        
        if self.num_features is None:
            self.num_features = X.select_dtypes(include='number').columns.to_list()
        
        self.scaler = StandardScaler()
        self.scaler.fit(X_standarded[self.num_features])
        X_standarded[self.num_features] = self.scaler.transform(X_standarded[self.num_features])
        
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_standarded[self.num_features])
        
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        
        if self.n_components is None:
            self.cumsum = np.cumsum(self.explained_variance_ratio_)
            
            self.n_components = np.argmax(self.cumsum >= self.optimum) + 1
            self.opt_variance = np.min(self.cumsum[self.n_components - 1])
            # переучиваем под лучшее число компонент
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(X_standarded[self.num_features])
            self.explained_variance_ratio_ = pca.explained_variance_ratio_
            
            
        return self
            

    def transform(self, X):
        X_standarded = X.copy()
        
        X_standarded[self.num_features] = self.scaler.transform(X[self.num_features])
        
        new_features = self.pca.transform(X_standarded[self.num_features])

        if self.return_df:
            columns = []
            for i in range(new_features.shape[1]):
                columns.append(f"PCA_{i}")
            df = pd.DataFrame(new_features, columns=columns)
            return df
        else:
            return new_features
        
    def plot_optimal(self):
        sns.set_theme(style="darkgrid")
        data = pd.DataFrame({'n_components': range(1, len(self.cumsum) + 1), 
                             'cumsum': self.cumsum.round(4)})
        sns.lineplot(data=data, x='n_components', y='cumsum', c='black')
        sns.scatterplot(data, x='n_components', y='cumsum', markers='o', c='black', s=50, alpha=1)
        plt.axvline(self.n_components, c='black', linestyle='--', alpha=0.3)
        plt.axhline(self.opt_variance, c='black', linestyle='--', alpha=0.3)
        plt.scatter(x=self.n_components, y=self.opt_variance, c='r', s=60, alpha=1)
        
        plt.xlabel("n_components")
        plt.ylabel("cumsum explained variance")
        plt.title("PCA - optimal n_components")

        plt.xticks(range(1, len(self.cumsum) + 1))
        
        plt.show()
        
# ==========================================================================================================================================================================

from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

class NMFCreator(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, num_features=None, return_df=False, max_iter=1000):
        self.n_components = n_components
        self.num_features = num_features
        self.return_df = return_df
        self.max_iter = max_iter
        
    def fit(self, X, y=None):
        X_minmaxed = X.copy()
        
        if self.num_features is None:
            self.num_features = X.select_dtypes(include='number').columns.to_list()
        
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_minmaxed[self.num_features])
        X_minmaxed[self.num_features] = self.scaler.transform(X_minmaxed[self.num_features])
        
        self.nmf = NMF(n_components=self.n_components, max_iter=self.max_iter)
        self.nmf.fit(X_minmaxed[self.num_features])
        
        return self
        
    def transform(self, X):
        X_minmaxed = X.copy()
        
        X_minmaxed[self.num_features] = self.scaler.transform(X[self.num_features])
        
        new_features = self.nmf.transform(X_minmaxed[self.num_features])

        if self.return_df:
            columns = []
            for i in range(new_features.shape[1]):
                columns.append(f"PCA_{i}")
            df = pd.DataFrame(new_features, columns=columns)
            return df
        else:
            return new_features