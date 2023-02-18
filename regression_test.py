#!/usr/bin/env python3

from pyDOE import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
import models_util as mu
from dvl_utils import generate_reference_points
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from problems_util import ProblemsUtil
import pandas as pd
from optproblems import dtlz
from optproblems import Individual
from sklearn.decomposition import PCA
import time
import os

def getMetrics(samples, problem_label, model_label, times, num_objective):
    print('Pipe {}'.format(model_label))
    problem = ProblemsUtil(problem_label, objectives=num_objective, variables=12)
    mses = np.zeros(times)
    maes = np.zeros(times)

    for i in range(times):
        start_time = time.time()
        solutions = lhs(problem.num_variables, samples=samples)
        objectives = np.array([problem.evaluate(solutions=sol) for sol in solutions])
        sol_test, obj_test = problem.optimal(samples=1000)

        if model_label == 'Linear':
            regressor = make_pipeline(LinearRegression())
        elif model_label == 'Ridge':
            regressor = make_pipeline(MultiOutputRegressor(Ridge(alpha=1.0)))
        elif model_label == 'RidgeCV':
            regressor = make_pipeline(MultiOutputRegressor(RidgeCV(alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75])))
        elif model_label == 'Polynomial':
            regressor = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        elif model_label == 'MLP':
            regressor = make_pipeline(MLPRegressor(hidden_layer_sizes=(11,11,11)))
        elif model_label == 'SVR':
            regressor = make_pipeline(MultiOutputRegressor(SVR(C=0.1, gamma="auto")))
        elif model_label == 'RFR':
            regressor = make_pipeline(RandomForestRegressor(n_estimators=100))
        if model_label == 'LinearSS':
            regressor = make_pipeline(StandardScaler(), LinearRegression())
        elif model_label == 'RidgeSS':
            regressor = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=1.0)))
        elif model_label == 'RidgeCVSS':
            regressor = make_pipeline(StandardScaler(), MultiOutputRegressor(RidgeCV(alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75])))
        elif model_label == 'PolynomialSS':
            regressor = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression())
        elif model_label == 'MLPSS':
            regressor = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(11,11,11)))
        elif model_label == 'SVRSS':
            regressor = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=0.1, gamma="auto")))
        elif model_label == 'RFRSS':
            regressor = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100))
        elif model_label == 'MLPCVSS':
            regressor = make_pipeline(StandardScaler(), MLPRegressor(activation='relu', solver='lbfgs', hidden_layer_sizes=(6,6,6), max_iter=5000))
        
        regressor.fit(objectives, solutions)

        #print(regressor.best_params_)    
        #print(regressor.best_score_)   
        #score = regressor.score(obj_test, sol_test)
        #print('Score {}'.format(score))

        previsoes = regressor.predict(obj_test)

        mse = mean_squared_error(sol_test, previsoes)
        #print('Mean Squared Error {}'.format(mse))
        mses[i] = mse
        
        diff = time.time() - start_time
        print('Execute {} in {} miliseconds '.format(i+1,diff))
        #mae = mean_absolute_error(sol_test, previsoes)
        #maes[i] = mae

    #print('Mean MSE {}'.format(np.mean(mses)))
    #print('STD MSE {}'.format(np.std(mses)))
    #print('Minimun MSE {}'.format(np.min(mses)))

    return mses, maes

def generateSpreadSheets(num_objectives):
    problem_labels = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7']
    scales = [250, 500, 1000, 1500, 10000]
    times = 20
    for problem_label in problem_labels:       
        for scale in scales:
            result = np.zeros((times,0))
            mses_linear, maes_linear = getMetrics(scale, problem_label, 'Linear', times, num_objectives)
            result = np.insert(result, result.shape[1], mses_linear, axis=1)
            mses_ridge, maes_ridge = getMetrics(scale, problem_label, 'Ridge', times, num_objectives)
            result = np.insert(result, result.shape[1], mses_ridge, axis=1)
            mses_poly, maes_poly = getMetrics(scale, problem_label, 'Polynomial', times, num_objectives)
            result = np.insert(result, result.shape[1], mses_poly, axis=1)
            mses_mlp, maes_mlp = getMetrics(scale, problem_label, 'MLP', times, num_objectives)
            result = np.insert(result, result.shape[1], mses_mlp, axis=1)
            mses_rfr, maes_rfr = getMetrics(scale, problem_label, 'RFR', times, num_objectives)
            result = np.insert(result, result.shape[1], mses_rfr, axis=1)
            mses_svr, maes_svr = getMetrics(scale, problem_label, 'SVR', times, num_objectives)
            result = np.insert(result, result.shape[1], mses_svr, axis=1)
            mses_linearSS, maes_linearSS = getMetrics(scale, problem_label, 'LinearSS', times, num_objectives)
            result = np.insert(result, result.shape[1], mses_linearSS, axis=1)
            mses_ridgeSS, maes_ridgeSS = getMetrics(scale, problem_label, 'RidgeSS', times, num_objectives)
            result = np.insert(result, result.shape[1], mses_ridgeSS, axis=1)
            mses_polySS, maes_polySS = getMetrics(scale, problem_label, 'PolynomialSS', times, num_objectives)
            result = np.insert(result, result.shape[1], mses_polySS, axis=1)
            mses_mlpSS, maes_mlpSS = getMetrics(scale, problem_label, 'MLPSS', times, num_objectives)
            result = np.insert(result, result.shape[1], mses_mlpSS, axis=1)
            mses_rfrSS, maes_rfrSS = getMetrics(scale, problem_label, 'RFRSS', times, num_objectives)
            result = np.insert(result, result.shape[1], mses_rfrSS, axis=1)
            mses_svrSS, maes_svrSS = getMetrics(scale, problem_label, 'SVRSS', times, num_objectives)
            result = np.insert(result, result.shape[1], mses_svrSS, axis=1)
            
            var2 = pd.DataFrame(result)
            
            if not os.path.exists('./Results'):
                # Create a new directory because it does not exist
                os.makedirs('./Results')
            var2.to_excel('./Results/regression_'+problem_label+'_'+str(num_objectives)+'_'+str(scale)+'.xlsx',header=False, index=False)

def generateBoxplot():
    num_objectives = [3,10]
    problem_labels = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7']
    scales = [1000]
    for num_objective in num_objectives:
        for problem_label in problem_labels:       
            for scale in scales:  
                data = pd.read_excel('./Results/VariableTest/vtest_'+problem_label+'_'+str(num_objective)+'_'+str(scale)+'.xlsx', header=None)   
                data = data.to_numpy()
        
                fig, ax = plt.subplots()
                ax.set_title(problem_label+'-'+str(num_objective)+' objectives.')
                ax.boxplot(data)
                ax.set_xticklabels(('SVR Ori.', 'SVR 1:1', 'MLP Ori.', 'MLP 1:1'))
                plt.savefig('./Results/VariableTest/BP_'+problem_label+'_'+str(num_objective)+'_'+str(scale)+'.png', transparent=False)


#generateSpreadSheets(3)