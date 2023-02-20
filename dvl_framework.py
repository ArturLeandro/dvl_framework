from pyDOE import *
from optproblems import dtlz
import numpy as np
import copy
import hvwfg
import pandas as pd
import time
from dvl_util import truncate, generate_reference_points
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.core.duplicate import NoDuplicateElimination
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler
from pymoo.factory import get_problem, get_reference_directions, get_sampling, get_crossover, get_mutation
from SUMOProblem import SUMOProblem
from pymoo.core.problem import Problem

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

class DVLFramework():

    def __init__(self,
                problem=None,
                pipeline=None,
                samples=500,
                evaluations=1500,
                num_process_time=None,
                num_order=None):

        if problem == None:
            raise Exception("Cannot initializate without a problem.")

        if pipeline == None:
            raise Exception("Cannot initializate without a regression pipeline.")

        self.sumo_problem = problem
        self.problem = self.sumo_problem.problem
        self.pipeline = pipeline
        self.samples = samples
        self.evaluations = evaluations
        self.num_process_time=num_process_time
        self.num_order=num_order
        self.hv_ref = self.problem.referenceHV()
        self.normal = np.prod(self.hv_ref)
        self.best_solutions = None
        self.best_objectives = None

        self.preprocessing = MaxAbsScaler()
    
    def execute(self):

        upper = np.ones(self.problem.num_variables)
        lower = np.zeros(self.problem.num_variables)

        solutions = lhs(self.problem.num_variables, samples=self.samples)
        objectives = np.array([self.problem.evaluate(sol) for sol in solutions])
        
        if self.problem.num_objectives == 10:
            reference_points = np.asarray(generate_reference_points(self.problem.num_objectives, 3))
        else:
            reference_points = np.asarray(generate_reference_points(self.problem.num_objectives, self.problem.num_variables))
        
        new_solutions = self.fit_pred_new_solutions(objectives, solutions, reference_points, lower, upper)    
        new_objectives = np.array([self.problem.evaluate(sol) for sol in new_solutions])
        
        solutions = np.concatenate((solutions, new_solutions))
        objectives = np.concatenate((objectives, new_objectives))

        self.best_solutions = new_solutions
        self.best_objectives = new_objectives

        best_hv = hvwfg.wfg(new_objectives, self.hv_ref)/self.normal
        print('DVL hypervolume {}'.format(best_hv))
        print('DVL evaluations {}'.format(objectives.shape[0]))
        diff = self.evaluations - (objectives.shape[0])
        if diff > 0:
            #algorithm = NSGA3(pop_size=reference_points.shape[0], ref_dirs=reference_points, sampling=np.asarray(self.best_solutions))
            algorithm = MOEAD(pop_size=reference_points.shape[0],ref_dirs=reference_points, sampling=np.asarray(self.best_solutions), eliminate_duplicates=NoDuplicateElimination(), n_neighbors=15,decomposition="pbi",prob_neighbor_mating=0.7,seed=1)
            pymoo_problem = get_problem(self.problem.problem_label.lower())
            pymoo_problem.n_obj = self.problem.num_objectives
            pymoo_problem.n_var = self.problem.num_variables
            pymoo_problem.xl = np.zeros(self.problem.num_variables)
            pymoo_problem.xu = np.ones(self.problem.num_variables)
            res = minimize(pymoo_problem,
                    algorithm,
                    seed=1,
                    termination=('n_eval', diff),
                    #termination=('n_gen', 300),
                    verbose=True)

            moea_objectives = np.array([self.problem.evaluate(sol) for sol in res.X])
            moea_hv = hvwfg.wfg(moea_objectives, self.hv_ref)/self.normal
            print('MOEA hypervolume {}'.format(moea_hv))
            if moea_hv > best_hv:
                best_hv = moea_hv
            print('DVL2 hypervolume {}'.format(best_hv))

        return best_hv

    def fit_pred_new_solutions(self, objectives, solutions, reference_points, lower, upper):
        new_solutions = []
        for rp in reference_points:
            self.pipeline.fit(objectives, solutions)
            rp_pred = self.pipeline.predict([rp])[0].tolist()
            new_solutions.append(truncate(rp_pred, lower, upper))

        return new_solutions

    def executeRealProblem(self):

        process_start = time.process_time()
        clock_start = time.time()

        upper = np.repeat(120, self.problem.num_variables)
        lower = np.repeat(20, self.problem.num_variables)

        solutions = (lhs(self.problem.num_variables, samples=self.samples) * 100) + 20
        solutions = solutions.astype(int)
        objectives = np.array([self.problem.evaluate(sol) for sol in solutions])
        
        if self.problem.num_objectives == 10 or self.problem.num_objectives == 6:
            reference_points = np.asarray(generate_reference_points(self.problem.num_objectives, 3))
        else:
            reference_points = np.asarray(generate_reference_points(self.problem.num_objectives, self.problem.num_variables))
        
        new_solutions = self.fit_pred_new_solutions_SUMO(objectives, solutions, reference_points, lower, upper)   
        new_array = [tuple(row) for row in new_solutions]
        new_solutions = np.unique(new_array, axis=0)
        new_objectives = np.array([self.problem.evaluate(sol) for sol in new_solutions])
        
        solutions = np.concatenate((solutions, new_solutions))
        objectives = np.concatenate((objectives, new_objectives))

        self.best_solutions = new_solutions
        self.best_objectives = new_objectives

        best_hv = hvwfg.wfg(new_objectives, self.hv_ref)/self.normal

        print('Total Processing Time. {}'.format(self.num_process_time))
        process_wasted = time.process_time() - process_start
        print('Wasted Time. {}'.format(process_wasted))
        print('DVL2 hypervolume {}'.format(best_hv))

        diff_time = self.num_process_time - process_wasted
        diff = int(diff_time / 0.22)
        print('Diff {}'.format(diff))

        if diff_time > 10:
            #algorithm = NSGA3(pop_size=reference_points.shape[0], ref_dirs=reference_points, sampling=np.asarray(self.best_solutions))
            #algorithm = NSGA2(pop_size=20,n_offsprings=20,sampling=np.asarray(self.best_solutions),crossover=get_crossover("int_sbx", prob=0.9, eta=15),mutation=get_mutation("int_pm", eta=20),eliminate_duplicates=True)
            algorithm = MOEAD(pop_size=reference_points.shape[0],ref_dirs=reference_points, sampling=np.asarray(self.best_solutions), eliminate_duplicates=NoDuplicateElimination(), n_neighbors=15,decomposition="pbi",prob_neighbor_mating=0.7,seed=1)
            res = minimize(self.sumo_problem,
                    algorithm,
                    seed=1,
                    termination=('n_eval', diff),
                    #termination=('n_gen', 300),
                    verbose=True)

            moea_objectives = np.array([self.problem.evaluate(sol) for sol in res.X])
            moea_hv = hvwfg.wfg(moea_objectives, self.hv_ref)/self.normal
            print('NSGA-2 hypervolume {}'.format(moea_hv))
            if moea_hv > best_hv:
                best_hv = moea_hv
                self.best_solutions = res.X
                self.best_objectives = moea_objectives

        process_final = time.process_time() - process_start
        clock_final = time.time() - clock_start
        print('Process Time {}'.format(process_final))    
        print('Clock Time {}'.format(clock_final))   
        with open('./Results/Experimento3/Bruto/time_'+str(self.num_process_time)+'_DVL2_'+str(self.num_order)+'.txt', 'w') as f:
            print('Process Time {}'.format(process_final), file=f)    
            print('Clock Time {}'.format(clock_final), file=f)   

        #Guardando o resultado do DVL
        backup = np.concatenate((self.best_solutions, self.best_objectives), axis=1)
        var2 = pd.DataFrame(backup)
        var2.to_excel('./Results/Experimento3/Bruto/result_'+str(self.num_process_time)+'_DVL2_'+str(self.num_order)+'.xlsx',header=False, index=False)

        print('Best hypervolume: {}'.format(best_hv))
        return best_hv

    def fit_pred_new_solutions_SUMO(self, objectives_not, solutions, reference_points, lower, upper):
        objectives = self.preprocessing.fit_transform(objectives_not)
        new_solutions = []
        for rp in reference_points:
            self.pipeline.fit(objectives, solutions)
            rp_pred = self.pipeline.predict([rp])[0].tolist()
            new_solutions.append(truncate([round(num) for num in rp_pred], lower, upper))

        return new_solutions