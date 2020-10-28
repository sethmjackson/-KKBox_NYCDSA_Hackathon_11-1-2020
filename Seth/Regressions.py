from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
import pandas as p
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd
import Seth.Util as ut
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List
import matplotlib.pyplot as plt
import time

import pickle


# d. regression

class Regression:
    def __init__(self, model, name: str, hyperparams={}):
        self.model = model
        self.name = name
        self.hyperparams = hyperparams
        self.bestParams: Dict
        self.time=''

    def fit(self, xTrain, xTest, yTrain, yTest):
        print('Starting ', self.name)

        self.model.fit(xTrain,yTrain)
        self.modelCV = self.model
        self.trainScore = self.model.score(xTrain, yTrain)
        self.testScore = self.model.score(xTest, yTest)
        #self.trainRMSE = self.getRMSE(yTrain, self.model.predict(xTrain))
        #self.testRMSE = self.getRMSE(yTest, self.model.predict(xTest))


    def fitCV(self,xTrain, xTest, yTrain, yTest, cv=5):
        print('Starting ', self.name)


        grid = RandomizedSearchCV(self.model, self.hyperparams, cv=cv, return_train_score = False,
                                  n_jobs=-1, n_iter=5, random_state=5)
        self.modelCV = grid.fit(xTrain,yTrain)
        self.bestParams = self.modelCV.best_params_
        self.trainScore = self.modelCV.best_estimator_.score(xTrain, yTrain)
        self.testScore = self.modelCV.best_estimator_.score(xTest, yTest)

        # if 'max_depth' in self.hyperparams:
        #     featureSelection = self.model.feature_importances_
        #     print(featureSelection)

        # self.trainRMSE = self.getRMSE(yTrain, self.modelCV.predict(xTrain))
        # self.testRMSE = self.getRMSE(yTest, self.modelCV.predict(xTest))


    def getRMSE(self, y, predicted):
        return sqrt(np.exp(mean_squared_error(y, predicted)))


    def plotHyperParams(self, trainX, testX, trainY, testY, i):
        for name, params in self.hyperparams.items():
            startTime = time.time()
            coefs = []
            intercepts = []
            trainScore = []
            testScore = []

            if len(params) < 2:
                continue

            for value in params:
                self.model.set_params(**{name: value})
                #print(name, '   Value: ',value)
                self.model.fit(trainX, trainY)
           # intercepts.append(self.model.intercept_)
           # coefs.append(self.model.coef_)
                trainScore.append(self.model.score(trainX, trainY))
                testScore.append(self.model.score(testX, testY))

            plt.plot(params, trainScore, label=r'train set $R^2$')
            plt.plot(params, testScore, label=r'test set $R^2$')

            plt.xlabel(name+' Value')
            plt.ylabel('R^2 Value')
            plt.title(self.name+' R^2 VS. '+ name)
            plt.legend(loc=4)
            plt.savefig('Output/Hyperparams/'+str(i)+' - '+self.name+' '+name+'.png')
            plt.clf()
            endTime = time.time()
            print(name + ': ' + ut.getTimeDiff(endTime - startTime))


def scaleData(df: pd.DataFrame, columns: List, inplace=False):
    if inplace:
        result = df
    else:
        result = df.copy()

    sc = StandardScaler()

    result[columns] = sc.fit_transform(result[columns])
    if inplace:
        return None
    else:
        return result

flatten = lambda l: [item for sublist in l for item in sublist]

def assembleModels():

    models = {

    # 'Random Forest': Regression(RandomForestClassifier(n_jobs=-1), 'Random Forest',
    # {   'max_depth': range(20, 40, 5),
    #     'n_estimators': range(30, 60, 4)}),

    'Gradient Boost': Regression(GradientBoostingClassifier(), 'Gradient Boost',
               {'learning_rate': np.linspace(.04, 0.7, 5),
                'n_estimators': range(80, 100, 5),
                'max_depth': range(10, 20, 3)}) # use feature_importances for feature selection

    # 'SVM': Regression(SVC(), 'Support Vector Regressor',
    #            {'C': np.linspace(1, 100, 10),
    #             'gamma': np.linspace(1e-7, 0.1, 10)}),

#

    #Regression((), ''),
    }
    return models

def performRegressions(df: pd.DataFrame, continuousColumns, outputColumn):
    print('Performing Regressions')
    models = assembleModels()
    y = df[outputColumn]
    x = scaleData(df.drop(columns=[outputColumn]), continuousColumns)


    trainTestData = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
    outputRatio = len(y[y > 0]) / len(y)

    #i=0
    # for name, model in models.items():
    #     print(name)
    #     model.plotHyperParams(*trainTestData,i)
    #     print('')
    #     i+=1

    for name in models.keys():
        models[name].time, returnValue = ut.getExecutionTime(lambda: models[name].fitCV(*trainTestData))

    results = pd.DataFrame([r.__dict__ for r in models.values()]).drop(columns=['model', 'modelCV'])

    roundColumns4Digits = ['trainScore', 'testScore']
    #roundColumns8Digits = ['trainRMSE', 'testRMSE']
    for c in roundColumns4Digits:
        results[c] = results[c].apply(ut.roundTraditional, args = (4,))

    results.to_excel('Output/Model Results.xlsx')
    print('Finished Regressions')
    return models





# def predictSalePrice(dfTest: pd.DataFrame, models: Dict):
#     continuousColumns = getColumnType(dfTest, 'Continuous', True)
#
#     x = scaleData(dfTest, continuousColumns)
#     predictions = pd.DataFrame()
#
#     for regression in models.values():
#         prediction = pd.Series(regression.model.predict(x))
#         predictions = ut.appendColumns([predictions, prediction])
#
#     salePrice  = predictions.apply(np.exp, axis=1)
#     finalPrediction = salePrice.apply(np.mean, axis=1)
#
#     output = pd.DataFrame({'Id': test['Id'].astype(int), 'SalePrice': finalPrediction})
#     output.to_csv('Output/Submission.csv', index=False)














# look up randomizedSearchCV vs. GridsearchCV

# use VIF > 5, AIC, BIC for feature selection
# Don't use linear regression on categorical vars
# create ensemble of many different models (check for packages that can do this)
# use linear model on everything, then feature select