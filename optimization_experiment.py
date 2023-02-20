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
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.moead import MOEAD
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import hvwfg
import matplotlib.pyplot as plt
import time

def plotObjSpace(objectives, reference_points, hv_point):
    xs = objectives[:,0]
    ys = objectives[:,1]
    zs = objectives[:,2]

    rxs = reference_points[:,0]
    rys = reference_points[:,1]
    rzs = reference_points[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(xs, ys, zs, color='blue')
    ax.scatter3D(rxs, rys, rzs, color='orange')
    ax.scatter3D(hv_point[0], hv_point[1], hv_point[2], color='red')

    ax.set_xlabel('Objective X')
    ax.set_ylabel('Objective Y')
    ax.set_zlabel('Objective Z')

    plt.show()


def runDVL(problem, model, iterations, samples, ref_dirs, hv_ref):
    pipeline=getModel(model)
    dvl = DVL(problem=problem, pipeline=pipeline, samples=samples, num_training=300, iterations=iterations)
    dvl.execute()
    best_solutions = np.asarray(dvl.best_solutions)
    print('Tamanho {}'.format(best_solutions.shape))    
    #plotObjSpace(best_solutions, ref_dirs, hv_ref)

def runDVLFramework(problem, model, samples, evaluations):
    pipeline=getModel(model)
    dvl = DVLFramework(problem=problem, pipeline=pipeline, samples=samples, evaluations=evaluations)
    dvl.execute()
    best_solutions = np.asarray(dvl.best_solutions)
    #print('Tamanho {}'.format(best_solutions.shape))    
    #plotObjSpace(best_solutions, ref_dirs, hv_ref)       

def runNSGA(problem, evaluations):  
    if problem.num_objectives == 3:
        ref_dirs = np.asarray(generate_reference_points(problem.num_objectives, problem.num_variables))
    else:
        ref_dirs = np.asarray(generate_reference_points(problem.num_objectives, 3))
    algorithm = NSGA3(pop_size=ref_dirs.shape[0], ref_dirs=ref_dirs)
    pymoo_problem = get_problem(problem.problem_label.lower())
    pymoo_problem.n_obj = problem.num_objectives
    pymoo_problem.n_var = problem.num_variables
    pymoo_problem.xl = np.zeros(problem.num_variables)
    pymoo_problem.xu = np.ones(problem.num_variables)

    res = minimize(pymoo_problem,
            algorithm,
            seed=1,
            termination=('n_eval', evaluations),
            #termination=('n_gen', 300),
            verbose=True)

    best_objectives = np.array([problem.evaluate(sol) for sol in res.X])
    #print(best_objectives)
    normal = np.prod(problem.referenceHV())
    hv = hvwfg.wfg(best_objectives, problem.referenceHV())
    nor_hv = hv/normal
    print('Hypervolume: {}'.format(nor_hv))
    return nor_hv


def runMOEAD(problem, evaluations):  
    if problem.num_objectives == 3:
        ref_dirs = np.asarray(generate_reference_points(problem.num_objectives, problem.num_variables))
    else:
        ref_dirs = np.asarray(generate_reference_points(problem.num_objectives, 3))
    algorithm = MOEAD(pop_size=ref_dirs.shape[0],
    ref_dirs=ref_dirs, 
    n_neighbors=15,
    decomposition="pbi",
    prob_neighbor_mating=0.7,
    seed=1)
    pymoo_problem = get_problem(problem.problem_label.lower())
    pymoo_problem.n_obj = problem.num_objectives
    pymoo_problem.n_var = problem.num_variables
    pymoo_problem.xl = np.zeros(problem.num_variables)
    pymoo_problem.xu = np.ones(problem.num_variables)

    res = minimize(pymoo_problem,
            algorithm,
            seed=1,
            termination=('n_eval', evaluations),
            #termination=('n_gen', 300),
            verbose=True)

    best_objectives = np.array([problem.evaluate(sol) for sol in res.X])
    #print(best_objectives)
    normal = np.prod(problem.referenceHV())
    hv = hvwfg.wfg(best_objectives, problem.referenceHV())
    nor_hv = hv/normal
    print('Hypervolume: {}'.format(nor_hv))
    return nor_hv

def saveXLSX(algorithm, result, problem_label, scale, num_objectives):    
    var2 = pd.DataFrame(result)
    var2.to_excel('./Results/Experimento2/'+str(algorithm)+'_'+problem_label+'_'+str(num_objectives)+'_'+str(scale)+'.xlsx',header=False, index=False)

def executeDVLNTimes(times, problem, pipeline, samples, iterations):
    mean_hv = np.zeros(times)
    for z in range(times):
        start_time = time.time()
        dvl = DVL(problem=problem, pipeline=pipeline, samples=samples, num_training=300, iterations=iterations)
        mean_hv[z] = dvl.execute()
        diff = time.time() - start_time
        print('Execute {} in {} miliseconds '.format(z+1,diff))
    
    return mean_hv

def generateDVLSpreadSheets(problem_labels, model_labels, scales, samples, iterations, num_objectives):
    times = 20

    for j,problem_label in  enumerate(problem_labels):
        print('Problem {}'.format(problem_label))
        problem = ProblemsUtil(problem_label, objectives=num_objectives, variables=12)
        for i,sample in enumerate(samples):
            result = np.zeros((times,0))
            print('DVL Model: {}, Iter:{}, Samples:{}'.format(model_labels[j], iterations[i], sample))
            result_linear = executeDVLNTimes(times, problem, getModel(model_labels[j]), sample, iterations[i])
            result = np.insert(result, result.shape[1], result_linear, axis=1)
            saveXLSX('DVL', result, problem_label, scales[i], num_objectives)
         
def executeNSGANTimes(times, problem, evaluations):
    mean_hv = np.zeros(times)
    for z in range(times):
        start_time = time.time()
        mean_hv[z] = runNSGA(problem, evaluations)
        diff = time.time() - start_time
        print('Execute {} in {} miliseconds '.format(z+1,diff))
    
    return mean_hv

def generateNSGASpreadSheets(problem_labels, evaluations, num_objectives):
    times = 20

    for problem_label in problem_labels:
        print('Problem {}'.format(problem_label))
        problem = ProblemsUtil(problem_label, objectives=num_objectives, variables=12)
        for evaluation in evaluations:
            result = np.zeros((times,0))
            print('NSGA-III: evaluations:{}'.format(evaluation))
            result_nsga = executeNSGANTimes(times, problem, evaluation)
            result = np.insert(result, result.shape[1], result_nsga, axis=1)
            saveXLSX('NSGA', result, problem_label, evaluation, num_objectives)

def executeMOEADNTimes(times, problem, evaluations):
    mean_hv = np.zeros(times)
    for z in range(times):
        start_time = time.time()
        mean_hv[z] = runMOEAD(problem, evaluations)
        diff = time.time() - start_time
        print('Execute {} in {} miliseconds '.format(z+1,diff))
    
    return mean_hv

def generateMOEADSpreadSheets(problem_labels, evaluations, num_objectives):
    times = 20

    for problem_label in problem_labels:
        print('Problem {}'.format(problem_label))
        problem = ProblemsUtil(problem_label, objectives=num_objectives, variables=12)
        for evaluation in evaluations:
            result = np.zeros((times,0))
            print('MOEAD: evaluations:{}'.format(evaluation))
            result_nsga = executeMOEADNTimes(times, problem, evaluation)
            result = np.insert(result, result.shape[1], result_nsga, axis=1)
            saveXLSX('MOEAD', result, problem_label, evaluation, num_objectives)

def executeDVLFrameworkNTimes(times, problem, pipeline, samples, evaluations):
    mean_hv = np.zeros(times)
    for z in range(times):
        start_time = time.time()
        dvl_framework = DVLFramework(problem=problem, pipeline=pipeline, samples=samples, evaluations=evaluations)
        mean_hv[z] = dvl_framework.execute()
        diff = time.time() - start_time
        print('Execute {} in {} miliseconds '.format(z+1,diff))
    
    return mean_hv

def generateDVLFrameworkSpreadSheets(problem_labels, model_labels, evaluations, samples, num_objectives):
    times = 20

    for j,problem_label in  enumerate(problem_labels):
        print('Problem {}'.format(problem_label))
        problem = ProblemsUtil(problem_label, objectives=num_objectives, variables=12)
        for i,evaluation in enumerate(evaluations):
            result = np.zeros((times,0))
            print('DVL Model: {}, Samples:{}, Eval:{}'.format(model_labels[j], evaluation, samples[i]))
            result_dvl = executeDVLFrameworkNTimes(times, problem, getModel(model_labels[j]), samples[i], evaluation)
            result = np.insert(result, result.shape[1], result_dvl, axis=1)
            saveXLSX('DVLFramework_MOEAD', result, problem_label, evaluation, num_objectives)

#num_objectives = 3

#generateDVLSpreadSheets(['DTLZ1'], ['SVR'], ['250', '500', '1000', '1500', '10000'], [50, 112, 132, 200, 300], [1, 2, 4, 6, 44], num_objectives)
#generateDVLSpreadSheets(['DTLZ2'], ['MLPSS'], ['250', '500', '1000', '1500', '10000'], [50, 112, 132, 200, 7388], [1, 2, 4, 6, 12], num_objectives)
#generateDVLSpreadSheets(['DTLZ3'], ['SVR'], ['250', '500', '1000', '1500', '10000'], [50, 112, 132, 200, 300], [1, 2, 4, 6, 44], num_objectives)
#generateDVLSpreadSheets(['DTLZ4'], ['MLPSS'], ['250', '500', '1000', '1500', '10000'], [50, 112, 132, 200, 7388], [1, 2, 4, 6, 12], num_objectives)
#generateDVLSpreadSheets(['DTLZ5'], ['MLPSS'], ['250', '500', '1000', '1500', '10000'], [50, 112, 132, 200, 7388], [1, 2, 4, 6, 12], num_objectives)
#generateDVLSpreadSheets(['DTLZ6'], ['MLPSS'], ['250', '500', '1000', '1500', '10000'], [50, 112, 132, 200, 7388], [1, 2, 4, 6, 12], num_objectives)
#generateDVLSpreadSheets(['DTLZ7'], ['MLPSS'], ['250', '500', '1000', '1500', '10000'], [50, 112, 132, 200, 7388], [1, 2, 4, 6, 12], num_objectives)
#generateDVLSpreadSheets(['DTLZ1'], ['SVR'], ['100000'], [300], [44], num_objectives)
#generateDVLSpreadSheets(['DTLZ2'], ['MLPSS'], ['100000'], [7388], [12], num_objectives)
#generateDVLSpreadSheets(['DTLZ3'], ['SVR'], ['100000'], [300], [44], num_objectives)
#generateDVLSpreadSheets(['DTLZ4'], ['MLPSS'], ['100000'], [7388], [12], num_objectives)
#generateDVLSpreadSheets(['DTLZ5'], ['MLPSS'], ['100000'], [7388], [12], num_objectives)
#generateDVLSpreadSheets(['DTLZ6'], ['MLPSS'], ['100000'], [7388], [12], num_objectives)
#generateDVLSpreadSheets(['DTLZ7'], ['MLPSS'], ['100000'], [7388], [12], num_objectives)

#num_objectives = 10
#generateNSGASpreadSheets(['DTLZ1'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateNSGASpreadSheets(['DTLZ2'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateNSGASpreadSheets(['DTLZ3'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateNSGASpreadSheets(['DTLZ4'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateNSGASpreadSheets(['DTLZ5'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateNSGASpreadSheets(['DTLZ6'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateNSGASpreadSheets(['DTLZ7'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateNSGASpreadSheets(['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7'], [100000], num_objectives)

#generateMOEADSpreadSheets(['DTLZ1'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateMOEADSpreadSheets(['DTLZ2'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateMOEADSpreadSheets(['DTLZ3'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateMOEADSpreadSheets(['DTLZ4'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateMOEADSpreadSheets(['DTLZ5'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateMOEADSpreadSheets(['DTLZ6'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateMOEADSpreadSheets(['DTLZ7'], [250, 500, 1000, 1500, 10000], num_objectives)
#generateMOEADSpreadSheets(['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7'], [100000], num_objectives)

#generateDVLFrameworkSpreadSheets(['DTLZ1'], ['SVR'], [250, 500, 1000, 1500, 10000], [159, 227, 250, 300, 300], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ2'], ['MLPSS'], [250, 500, 1000, 1500, 10000], [159, 227, 600, 600, 600], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ3'], ['SVR'], [250, 500, 1000, 1500, 10000], [159, 227, 300, 300, 300], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ4'], ['MLPSS'], [250, 500, 1000, 1500, 10000], [159, 410, 410, 410, 500], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ5'], ['MLPSS'], [250, 500, 1000, 1500, 10000], [159, 227, 300, 300, 300], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ6'], ['MLPSS'], [250, 500, 1000, 1500, 10000], [159, 227, 300, 600, 600], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ7'], ['MLPSS'], [250, 500, 1000, 1500, 10000], [159, 400, 600, 600, 600], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ1'], ['SVR'], [100000], [300], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ2'], ['MLPSS'], [100000], [600], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ3'], ['SVR'], [100000], [300], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ4'], ['MLPSS'], [100000], [500], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ5'], ['MLPSS'], [100000], [300], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ6'], ['MLPSS'], [100000], [600], 3)
#generateDVLFrameworkSpreadSheets(['DTLZ7'], ['MLPSS'], [100000], [600], 3)

#generateDVLFrameworkSpreadSheets(['DTLZ1'], ['SVR'], [250, 500, 1000, 1500, 10000], [50, 112, 200, 200, 300], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ2'], ['MLPSS'], [250, 500, 1000, 1500, 10000], [50, 280, 300, 300, 300], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ3'], ['SVR'], [250, 500, 1000, 1500, 10000], [50, 112, 200, 200, 300], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ4'], ['MLPSS'], [250, 500, 1000, 1500, 10000], [50, 280, 300, 300, 300], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ5'], ['MLPSS'], [250, 500, 1000, 1500, 10000], [50, 280, 300, 300, 600], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ6'], ['MLPSS'], [250, 500, 1000, 1500, 10000], [50, 100, 100, 100, 100], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ7'], ['MLPSS'], [250, 500, 1000, 1500, 10000], [50, 100, 200, 300, 600], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ1'], ['SVR'], [100000], [300], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ2'], ['MLPSS'], [100000], [300], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ3'], ['SVR'], [100000], [300], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ4'], ['MLPSS'], [100000], [300], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ5'], ['MLPSS'], [100000], [600], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ6'], ['MLPSS'], [100000], [100], 10)
#generateDVLFrameworkSpreadSheets(['DTLZ7'], ['MLPSS'], [100000], [600], 10)


## DVL, NSGA AND DVL FRAMEWORK PARAMETERS  ##
# DTLZ1 (1,1,1) 3 obj
# DTLZ1 250: DVL (1, 159) - NSGA (250) - DVL FRAMEWORK (159, 250)
# DTLZ1 500: DVL (3, 227) - NSGA (500) - DVL FRAMEWORK (227, 500)
# DTLZ1 1000: DVL (9, 250) - NSGA (1000) - DVL FRAMEWORK (250, 1000)
# DTLZ1 1500: DVL (14, 250) - NSGA (1500) - DVL FRAMEWORK (300, 1500)
# DTLZ1 10000: DVL (100, 300) - NSGA (10000) - DVL FRAMEWORK (300, 10000)
# DTLZ1 (5,5, ..., 5) 10 obj
# DTLZ1 250: DVL (1, 50) - NSGA (250) - DVL FRAMEWORK (50, 250)
# DTLZ1 500: DVL (2, 112) - NSGA (500) - DVL FRAMEWORK (112, 500)
# DTLZ1 1000: DVL (4, 132) - NSGA (1000) - DVL FRAMEWORK (200, 1000)
# DTLZ1 1500: DVL (6, 200) - NSGA (1500) - DVL FRAMEWORK (200, 1500)
# DTLZ1 10000: DVL (44, 300) - NSGA (10000) - DVL FRAMEWORK (300, 10000)

# DTLZ2 (2,2,2)
# DTLZ2 250: DVL (1, 159) - NSGA (250) - DVL FRAMEWORK (159, 250)
# DTLZ2 500: DVL (3, 227) - NSGA (500) - DVL FRAMEWORK (227, 500)
# DTLZ2 1000: DVL (8, 363) - NSGA (1000) - DVL FRAMEWORK (600, 1000)
# DTLZ2 1500: DVL (9, 681) - NSGA (1500) - DVDVL FRAMEWORKL2 (600, 1500)
# DTLZ2 10000: DVL (14, 8726) - NSGA (10000) - DVL FRAMEWORK (600, 10000)
# DTLZ2 (1.5,1.5, ..., 1.5) 10 obj
# DTLZ2 250: DVL (1, 50) - NSGA (250) - DVL FRAMEWORK (50, 250)
# DTLZ2 500: DVL (2, 112) - NSGA (500) - DVL FRAMEWORK (280, 500)
# DTLZ2 1000: DVL (4, 132) - NSGA (1000) - DVL FRAMEWORK (300, 1000)
# DTLZ2 1500: DVL (6, 200) - NSGA (1500) - DVL FRAMEWORK (300, 1500)
# DTLZ2 10000: DVL (12, 7388) - NSGA (10000) - DVL FRAMEWORK (300, 10000)

# DTLZ3 (10,10,10)
# DTLZ3 250: DVL (1, 159) - NSGA (250) - DVL FRAMEWORK (159, 250)
# DTLZ3 500: DVL (3, 227) - NSGA (500) - DVL FRAMEWORK (227, 500)
# DTLZ3 1000: DVL (8, 300) - NSGA (1000) - DVL FRAMEWORK (300, 1000)
# DTLZ3 1500: DVL (10, 300) - NSGA (1500) - DVL FRAMEWORK (300, 1500)
# DTLZ3 10000: DVL (20, 300) - NSGA (10000) - DVL FRAMEWORK (300, 10000)
# DTLZ3 (50,50, ..., 50) 10 obj
# DTLZ3 250: DVL (1, 50) - NSGA (250) - DVL FRAMEWORK (50, 250)
# DTLZ3 500: DVL (2, 112) - NSGA (500) - DVL FRAMEWORK (112, 500)
# DTLZ3 1000: DVL (4, 132) - NSGA (1000) - DVL FRAMEWORK (200, 1000)
# DTLZ3 1500: DVL (6, 200) - NSGA (1500) - DVL FRAMEWORK (200, 1500)
# DTLZ3 10000: DVL (44, 300) - NSGA (10000) - DVL FRAMEWORK (300, 10000)

# DTLZ4 (2,2,2)
# DTLZ4 250: DVL (1, 159) - NSGA (250) - DVL FRAMEWORK (159, 250)
# DTLZ4 500: DVL (3, 227) - NSGA (500) - DVL FRAMEWORK (410, 500)
# DTLZ4 1000: DVL (8, 300) - NSGA (1000) - DVL FRAMEWORK (410, 1000)
# DTLZ4 1500: DVL (9, 681) - NSGA (1500) - DVL FRAMEWORK (410, 1500)
# DTLZ4 10000: DVL (16, 681) - NSGA (10000) - DVL FRAMEWORK (500, 10000)
# DTLZ4 (1.5,1.5, ..., 1.5) 10 obj
# DTLZ4 250: DVL (1, 50) - NSGA (250) - DVL FRAMEWORK (50, 250)
# DTLZ4 500: DVL (2, 112) - NSGA (500) - DVL FRAMEWORK (280, 500)
# DTLZ4 1000: DVL (4, 132) - NSGA (1000) - DVL FRAMEWORK (300, 1000)
# DTLZ4 1500: DVL (6, 200) - NSGA (1500) - DVL FRAMEWORK (300, 1500)
# DTLZ4 10000: DVL (12, 7388) - NSGA (10000) - DVL FRAMEWORK (300, 10000)

# DTLZ5 (3,5,6)
# DTLZ5 250: DVL (1, 159) - NSGA (250) - DVL FRAMEWORK (159, 250)
# DTLZ5 500: DVL (3, 227) - NSGA (500) - DVL FRAMEWORK (227, 500)
# DTLZ5 1000: DVL (8, 300) - NSGA (1000) - DVL FRAMEWORK (300, 1000)
# DTLZ5 1500: DVL (9, 681) - NSGA (1500) - DVL FRAMEWORK (300, 1500)
# DTLZ5 10000: DVL (16, 681) - NSGA (10000) - DVL FRAMEWORK (300, 10000)
# DTLZ5 (4,4, ..., 4) 10 obj
# DTLZ5 250: DVL (1, 50) - NSGA (250) - DVL FRAMEWORK (50, 250)
# DTLZ5 500: DVL (2, 112) - NSGA (500) - DVL FRAMEWORK (280, 500)
# DTLZ5 1000: DVL (4, 132) - NSGA (1000) - DVL FRAMEWORK (300, 1000)
# DTLZ5 1500: DVL (6, 200) - NSGA (1500) - DVL FRAMEWORK (300, 1500)
# DTLZ5 10000: DVL (12, 7388) - NSGA (10000) - DVL FRAMEWORK (600, 10000)

# DTLZ6 (2 + (4 * (i + 1)))
# DTLZ6 250: DVL (1, 159) - NSGA (250) - DVL FRAMEWORK (159, 250)
# DTLZ6 500: DVL (3, 227) - NSGA (500) - DVL FRAMEWORK (227, 500)
# DTLZ6 1000: DVL (8, 300) - NSGA (1000) - DVL FRAMEWORK (300, 1000)
# DTLZ6 1500: DVL (9, 681) - NSGA (1500) - DVL FRAMEWORK (600, 1500)
# DTLZ6 10000: DVL (16, 681) - NSGA (10000) - DVL FRAMEWORK (600, 10000)
# DTLZ6 (2 + i) 10 obj
# DTLZ6 250: DVL (1, 50) - NSGA (250) - DVL FRAMEWORK (50, 250)
# DTLZ6 500: DVL (2, 112) - NSGA (500) - DVL FRAMEWORK (100, 500)
# DTLZ6 1000: DVL (4, 132) - NSGA (1000) - DVL FRAMEWORK (100, 1000)
# DTLZ6 1500: DVL (6, 200) - NSGA (1500) - DVL FRAMEWORK (100, 1500)
# DTLZ6 10000: DVL (12, 7388) - NSGA (10000) - DVL FRAMEWORK (100, 10000)

# DTLZ7 (2 + (4 * (i + 1)))
# DTLZ7 250: DVL (1, 159) - NSGA (250) - DVL FRAMEWORK (159, 250)
# DTLZ7 500: DVL (3, 227) - NSGA (500) - DVL FRAMEWORK (400, 500)
# DTLZ7 1000: DVL (8, 300) - NSGA (1000) - DVL FRAMEWORK (600, 1000)
# DTLZ7 1500: DVL (9, 681) - NSGA (1500) - DVL FRAMEWORK (600, 1500)
# DTLZ7 10000: DVL (16, 681) - NSGA (10000) - DVL FRAMEWORK (600, 10000)
# DTLZ7 (2 + (4 * (i + 1))) 10 obj
# DTLZ7 250: DVL (1, 50) - NSGA (250) - DVL FRAMEWORK (50, 250)
# DTLZ7 500: DVL (2, 112) - NSGA (500) - DVL FRAMEWORK (100, 500)
# DTLZ7 1000: DVL (4, 132) - NSGA (1000) - DVL FRAMEWORK (200, 1000)
# DTLZ7 1500: DVL (6, 200) - NSGA (1500) - DVL FRAMEWORK (300, 1500)
# DTLZ7 10000: DVL (12, 7388) - NSGA (10000) - DVL FRAMEWORK (600, 10000)