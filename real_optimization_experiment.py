#!/usr/bin/env python3

from numpy.core.fromnumeric import var
import numpy as np
import pandas as pd
from dvl import DVL
from dvl_framework import DVLFramework
from dvl_util import truncate, generate_reference_points
from models_util import getModel
from problems_util import ProblemsUtil
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, LabelBinarizer, Binarizer
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.factory import get_problem, get_reference_directions, get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import hvwfg
import matplotlib.pyplot as plt
import time
from SUMOProblem import SUMOProblem
import autograd.numpy as anp
from pymoo.core.problem import Problem
from pyDOE import *
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, mean_absolute_error


class SumoPymooProblem(Problem):

    def __init__(self):
        super().__init__(n_var=8,
                         n_obj=6,
                         n_constr=0,
                         xl=np.array([20, 20, 20, 20, 20, 20, 20, 20]),
                         xu=np.array([120, 120, 120, 120, 120, 120, 120, 120]),
                         type_var=int)
        
        self.problem = SUMOProblem(scenario=2)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.zeros((x.shape[0], self.n_obj))
        
        for i,dv in enumerate(x):
            out["F"][i] = self.problem.evaluate(dv) 

        #print(out["F"])

def runNSGA2(evaluations, order):

    problem = SumoPymooProblem()
    algorithm = NSGA2(
        pop_size=20,
        n_offsprings=20,
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_sbx", prob=0.9, eta=15),
        mutation=get_mutation("int_pm", eta=20),
        eliminate_duplicates=True
    )
    
    process_start = time.process_time()
    clock_start = time.time()
    res = minimize(problem,
                algorithm,
                seed=1,
                termination=('n_eval', evaluations),
                #termination=('n_gen', 300),
                verbose=True)

    process_final = time.process_time() - process_start
    clock_final = time.time() - clock_start
    print('Process Time {}'.format(process_final))    
    print('Clock Time {}'.format(clock_final))   
    with open('./Results/Experimento3/Bruto/time_'+str(evaluations)+'_NSGA2_'+str(order)+'.txt', 'w') as f:
        print('Process Time {}'.format(process_final), file=f)    
        print('Clock Time {}'.format(clock_final), file=f)   

    #Saving results
    result = np.concatenate((res.X, res.F), axis=1)
    var2 = pd.DataFrame(result)
    var2.to_excel('./Results/Experimento3/Bruto/result_'+str(evaluations)+'_NSGA2_'+str(order)+'.xlsx',header=False, index=False)


def generate_database():
    problem = SUMOProblem(scenario=2)
    solutions = (lhs(8, 500) * 100) + 20
    solutions = solutions.astype(int)
    objectives = np.zeros((solutions.shape[0], problem.num_objectives))
    for i,dv in enumerate(solutions):
        print('Evaluating {}'.format(i+1))
        process_start = time.process_time()
        clock_start = time.time()
        objectives[i] = problem.evaluate(dv) 
        print('Process Time {}'.format(time.process_time() - process_start))    
        print('Clock Time {}'.format(time.time() - clock_start))   
    
    result = np.concatenate((solutions, objectives), axis=1)
    var2 = pd.DataFrame(result)
    var2.to_excel('./Results/Experimento3/database500.xlsx',header=False, index=False)

def read_database():
    data = pd.read_excel('./Results/Experimento3/database500.xlsx', header=None)
    data_array = data.to_numpy()

    #data500 = pd.read_excel('./Results/Experimento3/database500.xlsx', header=None)
    #data500_array = data500.to_numpy()

    #data250 = pd.read_excel('./Results/Experimento3/database.xlsx', header=None)
    #data250_array = data250.to_numpy()

    #data_array = np.append(data250_array, data500_array, axis=0)

    return data_array[:,0:8], data_array[:,8:]

#@ignore_warnings(category=ConvergenceWarning)
def HPOptimization():
    y, X = read_database()

    #my_pipeline = getModel('RFRSS')
    #my_pipeline = make_pipeline(StandardScaler(), MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(11,11), activation='relu', solver='lbfgs', learning_rate='invscaling')))
    #my_pipeline = make_pipeline(Normalizer(), MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(3,), activation='tanh', solver='lbfgs', learning_rate='invscaling')))
    my_pipeline = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(11,11), activation='relu', solver='lbfgs', learning_rate='invscaling'))
    #my_pipeline = make_pipeline(MultiOutputRegressor(SVR(C=0.1, gamma="auto")))
    #my_pipeline = make_pipeline(MultiOutputRegressor(SVR(kernel='rbf', C=10, degree=3, coef0=0.01, gamma="auto")))
    #my_pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100))
    #my_pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(bootstrap=True,  max_depth=20, max_features='auto', min_samples_leaf=2, min_samples_split=10,  n_estimators=200))
    #kernel = 1.0 * RBF(1.0)
    #my_pipeline = make_pipeline(StandardScaler(), MultiOutputClassifier(GaussianProcessClassifier(kernel=kernel, random_state=0)))
    #param_grid = {'multioutputclassifier__estimator__activation': ['relu', 'tanh', 'logistic'],
    #      'multioutputclassifier__estimator__hidden_layer_sizes': [(3,), (3,3), (3,3,3), (5,), (5,5), (11,), (11, 11), (20,), (20,20), (20, 20, 20)],
    #      'multioutputclassifier__estimator__solver': ['lbfgs', 'sgd', 'adam'],
    #      'multioutputclassifier__estimator__learning_rate': ['constant', 'invscaling', 'adaptive'],
    #     }

    #param_grid = {'mlpregressor__activation': ['relu', 'tanh', 'logistic',],
    #      'mlpregressor__hidden_layer_sizes': [(3,), (3,3), (3,3,3), (5,), (5,5), (11,), (11, 11), (20,), (20,20), (20, 20, 20)],
    #     }
    #

    #param_grid = {'multioutputregressor__estimator__kernel' : ('rbf', 'sigmoid'),
    #    'multioutputregressor__estimator__C' : [0.1,1,5,10],
    #    'multioutputregressor__estimator__degree' : [3,8],
    #    'multioutputregressor__estimator__coef0' : [0.01,10,0.5],
    #    'multioutputregressor__estimator__gamma' : ('auto','scale')},

    #param_grid = {'randomforestregressor__n_estimators': [200, 600, 1000, 1400, 1800, 2000],
    #           'randomforestregressor__max_features': ['auto', 'sqrt'],
    #           'randomforestregressor__max_depth': [10, 20, 50, 80, 100, None],
    #           'randomforestregressor__min_samples_split': [2, 5, 10],
    #           'randomforestregressor__min_samples_leaf': [1, 2, 4],
    #           'randomforestregressor__bootstrap': [True, False]}

    #search = GridSearchCV(my_pipeline, param_grid, cv=5, n_jobs=-1, scoring = 'neg_mean_absolute_error', verbose=5)
    #search.fit(X, y)
    #print("Best parameter (CV score=%0.3f):" % search.best_score_)
    #print(search.best_params_)
    
    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    #scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    print("Average MSE score:", scores.mean())

    #my_pipeline.fit(X, y)
    ##print(X[0])
    ##print(y[0])
    #result = my_pipeline.predict([X[0]])
    #print(result)
    #print('MSE: {}'.format(mean_squared_error(y[0], result[0])))
    #print('MAE: {}'.format(mean_absolute_error(y[0], result[0])))

@ignore_warnings(category=ConvergenceWarning)
def executeDVLNTimes(times, problem, pipeline, samples, iterations):
    mean_hv = np.zeros(times)
    for z in range(times):
        start_time = time.time()
        dvl = DVL(problem=problem, pipeline=pipeline, samples=samples, num_training=300, iterations=iterations)
        mean_hv[z] = dvl.executeRealProblem()
        diff = time.time() - start_time
        print('Execute {} in {} miliseconds '.format(z+1,diff))
    
    return mean_hv


def testDVL(samples, iterations):
    process_start = time.process_time()
    clock_start = time.time()
    problem = SUMOProblem(scenario=2)
    my_pipeline = make_pipeline(MLPRegressor(hidden_layer_sizes=(11,11), activation='relu', solver='lbfgs', learning_rate='invscaling'))
    print(executeDVLNTimes(1, problem, my_pipeline, samples, iterations))
    
    process_final = time.process_time() - process_start
    clock_final = time.time() - clock_start
    print('Process Time {}'.format(process_final)) 
    print('Clock Time {}'.format(clock_final))   
    with open('./Results/Experimento3/time_'+str(samples)+'_'+str(iterations)+'_dvl_norm100.txt', 'w') as f:
        print('Process Time {}'.format(process_final), file=f)    
        print('Clock Time {}'.format(clock_final), file=f)   


def test_preprocessing():
    y, X = read_database()
    #my_pipeline = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(11,11), activation='relu', solver='lbfgs', learning_rate='invscaling'))
    print(X[0])
    X_ss = StandardScaler().fit_transform(X)
    print(X_ss[0])
    X_mms = MinMaxScaler().fit_transform(X)
    print(X_mms[0])
    X_mas = MaxAbsScaler().fit_transform(X)
    print(X_mas[0])
    X_norm = Normalizer().fit_transform(X)
    print(X_norm[0])

def test_hv():
    problem = SUMOProblem(scenario=2)
    hv_ref = problem.referenceHV()
    normal = np.prod(hv_ref)

    nsg2_data = pd.read_excel('./Results/Experimento3/result_300_NSGA2.xlsx', header=None)
    nsg2_data_array = nsg2_data.to_numpy()
    nsg2_objs = nsg2_data_array[:,8:].copy(order='C')
    nsg2_hv = hvwfg.wfg(nsg2_objs, hv_ref)
    print('NSGA2 300 hypervolume: {}'.format(nsg2_hv))
    nsg2_hv_nor = nsg2_hv/normal
    print('NSGA2 300 hypervolume Norm.: {}'.format(nsg2_hv_nor))

    
    #nsg2_500_data = pd.read_excel('./Results/Experimento3/result_1000_NSGA2.xlsx', header=None)
    #nsg2_500_data_array = nsg2_500_data.to_numpy()
    #nsg2_500_objs = nsg2_500_data_array[:,8:].copy(order='C')
    #nsg2_500_hv = hvwfg.wfg(nsg2_500_objs, hv_ref)
    #print('NSGA2 500 hypervolume: {}'.format(nsg2_500_hv))
    #nsg2_500_hv_nor = nsg2_500_hv/normal
    #print('NSGA2 500 hypervolume Norm.: {}'.format(nsg2_500_hv_nor))


    dvl_data = pd.read_excel('./Results/Experimento3/bruto/result_5000_NSGA2_1.xlsx', header=None)
    dvl_data_array = dvl_data.to_numpy()
    dvl_objs = dvl_data_array[:,8:].copy(order='C')
    dvl_hv = hvwfg.wfg(dvl_objs, hv_ref)
    print('DVL hypervolume: {}'.format(dvl_hv))
    dvl_hv_nor = dvl_hv/normal
    print('DVL hypervolume Norm.: {}'.format(dvl_hv_nor))

    dvl_norm_data = pd.read_excel('./Results/Experimento3/bruto/result_5000_NSGA2_12.xlsx', header=None)
    dvl_norm_data_array = dvl_norm_data.to_numpy()
    dvl_norm_objs = dvl_norm_data_array[:,8:].copy(order='C')
    dvl_norm_hv = hvwfg.wfg(dvl_norm_objs, hv_ref)
    print('DVL Norm. hypervolume: {}'.format(dvl_norm_hv))
    dvl_norm_hv_nor = dvl_norm_hv/normal
    print('DVL Norm. hypervolume Norm.: {}'.format(dvl_norm_hv_nor))

    
    #dvl_norm_x100_data = pd.read_excel('./Results/Experimento3/backup_dvl_Normalizerx100_real_250_5_279.xlsx', header=None)
    #dvl_norm_x100_data_array = dvl_norm_x100_data.to_numpy()
    #dvl_norm_x100_objs = dvl_norm_x100_data_array[:,8:].copy(order='C')
    #dvl_norm_x100_hv = hvwfg.wfg(dvl_norm_x100_objs, hv_ref)
    #print('DVL Norm. hypervolume: {}'.format(dvl_norm_x100_hv))
    #dvl_norm_x100_hv_nor = dvl_norm_x100_hv/normal
    #print('DVL Norm. hypervolume Norm.: {}'.format(dvl_norm_x100_hv_nor))


def test_hv_loop():
    problem = SUMOProblem(scenario=2)
    hv_ref = problem.referenceHV()
    normal = np.prod(hv_ref)

    for x in range(1,21):
        result = pd.read_excel('./Results/Experimento3/bruto/result_70_DVL2_'+str(x)+'.xlsx', header=None)
        result_array = result.to_numpy()
        result_objs = result_array[:,8:].copy(order='C')
        result_hv = hvwfg.wfg(result_objs, hv_ref)
        print('DVL2 {}: {}'.format(x, result_hv))
        result_hv_nor = result_hv/normal
        print('DVL2 Norm.  {}: {}'.format(x, result_hv_nor))

def transform_database():
    data = pd.read_excel('./Results/Experimento3/database.xlsx', header=None)
    data_array = data.to_numpy()

    #print(data_array[:,12])
    
    data_array[:,12] =  data_array[:,12]/1000

    var2 = pd.DataFrame(data_array)
    var2.to_excel('./Results/Experimento3/database_new.xlsx',header=False, index=False)

@ignore_warnings(category=ConvergenceWarning)
def runDVL(time_evaluation, order):    

    problem = SUMOProblem(scenario=2)
    my_pipeline = make_pipeline(MLPRegressor(hidden_layer_sizes=(11,11), activation='relu', solver='lbfgs', learning_rate='invscaling'))
    dvl = DVL(problem=problem, pipeline=my_pipeline, samples=250, num_training=300, num_process_time=time_evaluation, num_order=order)
    dvl.executeRealProblem()
    return    

@ignore_warnings(category=ConvergenceWarning)
def runDVLFramework(time_evaluation, order):    
    problem = SumoPymooProblem()
    #problem = SUMOProblem(scenario=2)
    my_pipeline = make_pipeline(MLPRegressor(hidden_layer_sizes=(11,11), activation='relu', solver='lbfgs', learning_rate='invscaling'))
    dvl_framework = DVLFramework(problem=problem, pipeline=my_pipeline, samples=250,  num_process_time=time_evaluation, num_order=order)
    dvl_framework.executeRealProblem()
    return    

#for x in range(1,21):
#  print('Executing {}...'.format(x))    
#  runNSGA2(5000, x) 


#for x in range(1,21):
#    print('Executing {}...'.format(x))    
#    runDVL(1100, x) 

#for x in range(1,21):
#    print('Executing {}...'.format(x))  
#    runDVLFramework(1100, x) 
