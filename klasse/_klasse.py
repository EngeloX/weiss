import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap 
import matplotlib.patheffects as pe
from cycler import cycler

from sklearn.base import BaseEstimator, TransformerMixin, clone

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, auc, roc_curve


GRUVBOX_COLORS = ["#c21161", "#ffec44"]

RC = {
    "figure.facecolor": "#1e222d", # задний фон фигуры
    "axes.facecolor": "#171b26", # фон графиков
    "grid.color": "#b4c2be", # цвет сетки
    "grid.alpha": 0.25, # прозрачность сетки
    "axes.edgecolor": "#e0e0e0", # Рамка
    "axes.linewidth": 0.8, # толщина рамки
    "legend.facecolor": "#323740", # цвет легенды
    "legend.labelcolor" : "ffffff", # Цвет текста на легендах
    "axes.titlecolor": "#ffffff", # Цвет текста в заголовке 
    "axes.labelcolor": "#ffffff", # мини заголовки осей
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    "xtick.color": "#ffffff",
    "ytick.color": "#ffffff",
    "axes.prop_cycle": cycler(color=GRUVBOX_COLORS),
}

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_features=None):
        self.cat_features = cat_features
        
    def fit(self, X, y=None):
        self.cat_features_ = self.cat_features or X.select_dtypes(include=['object', 'bool'])

        self.is_fitted_ = True 
        return self
        
    def transform(self, X):
        X_out = X.copy()
        for col in self.cat_features_:
            X_out[col] = X_out[col].astype('category')
        return X_out
    
    
class Trainer:
    def __init__(self, estimators, final_estimator=None, cv=None, cv_meta=None, seed=None, verbose=False):
        self.estimators = estimators
        self.final_estimator = final_estimator or LogisticRegression()
        self.seed = seed
        self.cv = cv or StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        self.cv_meta = cv_meta or StratifiedKFold(n_splits=10, shuffle=True, 
                                                   random_state=(self.seed*2 if self.seed is not None else self.seed))
        self.verbose = verbose
    def train(self, X, y):
        self.__X, self.__y = X.copy(), y.copy()
        
        self.estimators_, self.oof = self.__train_base_models(self.__X, self.__y, self.estimators)
        self.final_estimator_, self.oof_meta = self.__train_meta_model(self.oof, self.__y)
        
    def evaluate(self, X, proba=True):
        df_preds = pd.DataFrame()
        for name, estimator in self.estimators_:
            y_pred_proba = estimator.predict_proba(X)[:, 1]
            df_preds[name] = y_pred_proba
        res = self.final_estimator_.predict_proba(df_preds)[:, 1]
        if proba:
            return res
        else:
            return (res>0.5).astype(int)
        
    def add(self, estimators):
        new_estimators, new_oof = self.__train_base_models(self.__X, self.__y, estimators)
        
        self.estimators_ += new_estimators
        self.oof = self.oof.join(new_oof)

        self.final_estimator_, self.oof_meta = self.__train_meta_model(self.oof, self.__y)
        
    def remove(self, estimator_name):
        self.oof = self.oof.drop(estimator_name, axis=1)
        self.estimators_ = [(name, est) for name, est in self.estimators_ if name != estimator_name]
        self.final_estimator_, self.oof_meta = self.__train_meta_model(self.oof, self.__y)
        
    def __train_base_models(self, X, y, estimators):
        oof_proba = np.zeros( (X.shape[0], len(estimators)), dtype=float )
        trained_estimators = []
        for i, (name, estimator) in enumerate(estimators):
            for n_fold, (train_id, valid_id) in enumerate(self.cv.split(X, y)):
                X_train = X.iloc[train_id].copy()
                y_train = y.iloc[train_id]
                X_valid = X.iloc[valid_id].copy()
                y_valid = y.iloc[valid_id]
                
                estimator_ = clone(estimator)
                estimator_.fit(X_train, y_train)
                
                y_pred_proba = estimator_.predict_proba(X_valid)[:, 1]
                oof_proba[valid_id, i] = y_pred_proba

                if self.verbose:
                    print(f'[{name}] training fold {n_fold+1} | roc_auc: {roc_auc_score(y_valid, y_pred_proba):.4f}') 
            if self.verbose:
                print('------')
                print(f'[{name}] OOF roc_auc: {roc_auc_score(self.__y, oof_proba[:, i]):.4f}')
                print(f'[{name}] retraining on full dataset') 
                print('________________________________________') 
                
            estimator_ = clone(estimator)
            estimator_.fit(X, y)
            trained_estimators.append( (name, estimator_) )
        oof = pd.DataFrame(oof_proba, columns=[estimators[i][0] for i in range(len(estimators))])
        return trained_estimators, oof
        
    def __train_meta_model(self, X,  y):
        meta_oof_proba = np.zeros(X.shape[0], dtype=float)
        for n_fold, (train_id, valid_id) in enumerate(self.cv_meta.split(X, y)):
            X_train = X.iloc[train_id].copy()
            y_train = y.iloc[train_id]
            X_valid = X.iloc[valid_id].copy()
            y_valid = y.iloc[valid_id]

            estimator_ = clone(self.final_estimator)
            estimator_.fit(X_train, y_train)
            meta_oof_proba[valid_id] = estimator_.predict_proba(X_valid)[:, 1]
            
        final_estimator_ = clone(self.final_estimator)
        final_estimator_.fit(X, y)
        meta_oof = pd.Series(meta_oof_proba, name='meta_model')

        if self.verbose:
            print(f'[Ensemble] OOF roc_auc: {roc_auc_score(self.__y, meta_oof):.4f}') # verbose
        
        return final_estimator_, meta_oof
        
    
    def plot_evaluation(self, test=None):
        with sns.axes_style("darkgrid", rc=RC):
            with plt.rc_context(RC):    
                
                plt.figure(figsize=(8, 6))
                sns.kdeplot(self.oof_meta, fill=True, label='OOF Predictions (Train)', alpha=0.25)
                if test is not None:
                    sns.kdeplot(self.evaluate(test), fill=True, label='Test Predictions', alpha=0.25)
                plt.title('Distribution of Predictions: OOF vs Test')
                plt.xlabel('Predicted Probability')
                plt.ylabel('Density')
                plt.legend()
                plt.show()
 
    def plot_score(self, figsize=None):
        with sns.axes_style("darkgrid"):
            with plt.rc_context(RC):
                
                figsize = figsize or (8, 6)
                base_scores = self.oof.apply(lambda col: roc_auc_score(self.__y, col))
                meta_score = roc_auc_score(self.__y, self.oof_meta)
                all_values = base_scores.to_list() + [meta_score]
                right_limit = max(all_values) + max(all_values) * 0.1
            
                fig, ax = plt.subplots(figsize=figsize)
                ax.barh(base_scores.index, base_scores.values, color='#ffec44', alpha=0.75)
                ax.barh('ensemble', meta_score, color='#c21161', alpha=0.75)
                ax.set_xlim((0, right_limit))
                ax.set_xlabel("ROC-AUC")
                for container in ax.containers:
                    ax.bar_label(
                        container,
                        fmt="%.4f",
                        color='white',
                        fontsize=11,
                        padding=5,
                        weight='bold',
                        label_type='edge',
                        path_effects=[
                            pe.Stroke(linewidth=0.9, foreground='black'),
                            pe.Normal()
                        ]
                    )
            
                ax.set_title("ROC-AUC of Base Models and Ensemble")
                plt.show()
        
    def plot_curve(self):
        with sns.axes_style("darkgrid", rc=RC):
            with plt.rc_context(RC):    
                
                cmap = LinearSegmentedColormap.from_list("my_cmap", ["#c21161", '#e0e0e0' ,"#ffec44"])
                plt.figure(figsize=(8, 6))
                for i, col in enumerate(self.oof.columns): 
                    color = cmap(i / (len(self.oof.columns) if len(self.oof.columns) > 1 else 1.0))
                    fpr, tpr, _ = roc_curve(self.__y, self.oof[col]) 
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label = f'{col}: {roc_auc:.4f}', color=color)
                fpr, tpr, _ = roc_curve(self.__y, self.oof_meta) 
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'ensemble: {roc_auc:.4f}', color='green', linewidth=2)
                plt.plot([0, 1], [0, 1],'--', color='white', linewidth=2.2)
                
                plt.title('Receiver Operating Characteristic') 
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.legend(loc = 'lower right')
                plt.xlim((0, 1))
                plt.ylim((0, 1))
                plt.show()