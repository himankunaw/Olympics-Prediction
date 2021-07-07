import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR, SVR

class Regressor:
    
    def __init__(self, model_type):
        
        self.model_type = model_type
        
        if model_type == 'Baseline':
            self.model = LinearRegression()
        elif model_type == 'LinearRegression':
            self.model = LinearRegression()
        elif model_type == 'Ridge':
            self.model = Ridge(alpha=0.5)
        elif model_type == 'Lasso':
            self.model = Lasso(alpha=0.5)
        elif model_type == 'SVR':
            self.model = LinearSVR(loss='squared_epsilon_insensitive', dual=False)
        elif model_type == "Poisson":
            self.model = PoissonRegressor(alpha=0.5)
        elif model_type == 'RandomForest':
            self.model = RandomForestRegressor()
        else:
            raise Exception('Model does not exist in Regressor class')
            
    def fit(self, x, y):
        self.model.fit(x,y)
        
    def predict(self, x):
        self.model.predict(x)
        
    # Cross Validation 
    
    def fit_cv(self, x,y):
        
        if self.model_type == 'LinearRegression':
            self.model_cv = self.model
            self.model_cv.fit(x,y)
            
        elif self.model_type == 'Baseline':
            self.model_cv = self.model
            self.model_cv.fit(x,y)
            
        elif self.model_type == 'Ridge':
            self.model_cv = RidgeCV(alphas=(np.linspace(0.01, 10.0, num=30)),
                                    fit_intercept=True).fit(x,y)
            
        elif self.model_type == 'Lasso':
            self.model_cv = LassoCV(alphas=np.linspace(0.01,10.0, 100)).fit(x,y)
            
                                    
        elif self.model_type == 'SVR':
            params = {'kernel':['poly','rbf','linear','sigmoid'],
                       'C':np.logspace(0,5,num=40),
                       'gamma':['scale']}

            
            self.model_cv = GridSearchCV(estimator=SVR(), 
                                      param_grid=params, 
                                      scoring='neg_mean_squared_error').fit(x,y)
            
        elif self.model_type == 'Poisson':
            params = {'alpha':np.linspace(0.01,10,100)}
            
            self.model_cv = GridSearchCV(estimator=self.model, 
                                     param_grid=params, 
                                     scoring='neg_mean_squared_error').fit(x,y)
            
        elif self.model_type == 'RandomForest':
            params = {'bootstrap': [True, False],
                      'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                      'max_features': ['auto', 'sqrt'],
                      'min_samples_leaf': [1, 2, 3, 4],
                      'min_samples_split': [2, 4, 6, 8, 10],
                      'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
            
            self.model_cv = RandomizedSearchCV(estimator = self.model, 
                                               param_distributions = params, 
                                               n_iter = 200, cv = 3, 
                                               random_state=42, n_jobs = -1).fit(x,y)
