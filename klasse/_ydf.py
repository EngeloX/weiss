import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import ydf


class YDFRegressor(BaseEstimator, RegressorMixin):
    """
    Обертка для GradientBoostedTreesLearner под регрессию
    """
    def __init__(self, **learner_params):
        self._estimator_type = "regressor"
        self.learner_params = learner_params

    def __create_df(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])        

        if y is not None:
            if isinstance(y, pd.Series):
                self.label_name_ = y.name
            df['label'] = y
        
        return df
        
    def fit(self, X, y):
        df = self.__create_df(X, y)
        
        ydf.verbose(0)                  
        self.model_ = ydf.GradientBoostedTreesLearner(label='label', task=ydf.Task.REGRESSION, **self.learner_params).train(df)
        return self
    
    def predict(self, X):
        df = self.__create_df(X)
        
        preds = self.model_.predict(df)
        return preds
        
    def evaluate(self, X, y):
        df = self.__create_df(X, y)
            
        self.evaluation_ = self.model_.evaluate(df)
        self.rmse_ = self.evaluation_.rmse
        
        return self.evaluation_
        
    def plot_tree(self):
        return self.model_.plot_tree()
        
    def describe(self):
        return self.model_.describe()
        
    def score(self, X, y):
        self.evaluate(X, y)
        return -self.rmse_
        
    def get_params(self,deep=True):
        return self.learner_params.copy()

    def set_params(self, **params):
        self.learner_params.update(params) 
        return self

    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

class YDFClassifier(BaseEstimator, ClassifierMixin):
    """
    Обертка для GradientBoostedTreesLearner под классификацию
    """
    def __init__(self, **learner_params):
        self._estimator_type = "classifier"
        self.learner_params = learner_params

    def __create_df(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])        

        if y is not None:
            if isinstance(y, pd.Series):
                self.label_name_ = y.name
            df['label'] = y
        
        return df
        
    def fit(self, X, y):
        df = self.__create_df(X, y)
        
        ydf.verbose(0)                  
        self.model_ = ydf.GradientBoostedTreesLearner(label='label', task=ydf.Task.CLASSIFICATION, **self.learner_params).train(df)
        self.classes_ = np.unique(df['label'])
        self.is_binary_ = len(self.classes_) == 2
        return self
    
    def predict(self, X):
        df = self.__create_df(X)
        preds = self.model_.predict(df)

        if self.is_binary_:
            return np.where(np.array(preds) >= 0.5, self.classes_[1], self.classes_[0])
        else:
            return np.array([max(p, key=p.get) for p in preds])
    
    def predict_proba(self, X):
        df = self.__create_df(X)
        preds = self.model_.predict(df)

        if self.is_binary_:
            probs = np.array(preds).reshape(-1, 1)
            return np.hstack([1 - probs, probs])
        else:
            probas = []
            for p in preds:
                probas.append([p.get(c, 0.0) for c in self.classes_])
            return np.array(probas)
        
    def evaluate(self, X, y):
        df = self.__create_df(X, y)
        self.evaluation_ = self.model_.evaluate(df)
        self.accuracy_ = self.evaluation_.accuracy
        return self.evaluation_
        
    def plot_tree(self):
        return self.model_.plot_tree()
        
    def describe(self):
        return self.model_.describe()
        
    def score(self, X, y):
        self.evaluate(X, y)
        return self.accuracy_
        
    def get_params(self, deep=True):
        return self.learner_params.copy()

    def set_params(self, **params):
        self.learner_params.update(params) 
        return self