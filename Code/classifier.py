import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import itertools

class Classifier:
    
    def __init__(self, model_type):
        
        self.model_type = model_type
        
        if model_type == 'Baseline':
            self.model = LogisticRegression(dual=False)
        elif model_type == 'LogisticRegression':
            self.model = LogisticRegression(dual=False)
        elif model_type == 'SVC':
            self.model = LinearSVC(dual=False)
        elif model_type == 'RandomForest':
            self.model = RandomForestClassifier(max_depth=5, n_estimators=20)
        elif model_type == 'MLP':
            self.model = MLPClassifier(alpha=0.3, max_iter=1000)
        elif model_type == 'AdaBoost':
            self.model = AdaBoostClassifier()
        elif model_type == 'GaussianNB':
            self.model = GaussianNB()
        else:
            raise Exception('Model does not exist in Classifier class')
            
    def fit(self,x,y):
        self.model.fit(x,y)
    
    def predict(self,x):
        self.model.predict(x)
        
    def fit_cv(self,x,y):
        
        if self.model_type == 'LogisticRegression':
            self.model_cv = LogisticRegressionCV(Cs=np.logspace(5,-5,num=100),
                                                 n_jobs=-1).fit(x, y)
        
        elif self.model_type == 'Baseline':
            self.model_cv = self.model.fit(x,y)
            
        elif self.model_type == 'SVC':
            params = {'kernel':['poly','linear'],
                      'C':np.logspace(0,5.0,num=40),
                      'gamma':['scale'],
                      'max_iter':[-1]}
            self.model_cv = GridSearchCV(estimator=SVC(),
                                         param_grid = params, cv=3,
                                         scoring='accuracy', n_jobs=-1).fit(x,y)
              

              
            
        elif self.model_type == 'MLP':
            params={'activation':['logistic','relu','tanh'],
                    'hidden_layer_sizes':[x for x in itertools.product((10,30,50),repeat=3)] + [(100,),(100,100,)],
                    'alpha':np.logspace(2,-5,num=100),
                    'learning_rate':['constant','invscaling','adaptive']}
            
            
            self.model_cv = RandomizedSearchCV(estimator = self.model, 
                                               param_distributions = params, 
                                               n_iter = 200, cv = 3, verbose=1, 
                                               random_state=42, n_jobs = -1,
                                               scoring='accuracy').fit(x, y)
            
        elif self.model_type == 'AdaBoost':
            params = {'learning_rate': np.linspace(0,50,10),
                      'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
            
            self.model_cv = RandomizedSearchCV(estimator = self.model, 
                                               param_distributions = params, 
                                               n_iter = 200, cv = 3, verbose=1, 
                                               random_state=42, n_jobs = -1,
                                               scoring='accuracy').fit(x, y)
            
        elif self.model_type == 'GaussianNB':
            params = {'var_smoothing': np.logspace(0,-9, num=100)}
            
            self.model_cv = GridSearchCV(estimator=self.model, 
                                         param_grid=params, verbose=1, 
                                         scoring='accuracy').fit(x, y)
            
        elif self.model_type == 'RandomForest':
            params = {'bootstrap': [True, False],
                      'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                      'max_features': ['auto', 'sqrt'],
                      'min_samples_leaf': [1, 2, 4],
                      'min_samples_split': [2, 5, 10],
                      'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
            
            self.model_cv = RandomizedSearchCV(estimator = self.model, 
                                               param_distributions = params, 
                                               n_iter = 200, cv = 3, verbose=1, 
                                               random_state=42, n_jobs = -1,
                                               scoring='accuracy').fit(x,y)