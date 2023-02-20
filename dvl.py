from pyDOE import *
from optproblems import dtlz
import numpy as np
import copy
import hvwfg
from dvl_util import truncate, generate_reference_points
import pandas as pd
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler
import time

class DVL():

    def __init__(self,
                problem=None,
                pipeline=None,
                pipelines=None,
                samples=500,
                num_training=300,
                iterations=10,
                num_process_time=None,
                num_order=None):

        if problem == None:
            raise Exception("Cannot initializate without a problem.")

        if pipeline == None:
            raise Exception("Cannot initializate without a regression pipeline.")

        self.problem = problem
        self.pipeline = pipeline
        self.samples = samples
        self.num_training = num_training
        self.hv_ref = self.problem.referenceHV()
        self.iterations = iterations
        self.num_process_time=num_process_time
        self.num_order=num_order
        self.normal = np.prod(self.hv_ref)
        self.diff = 0.0001
        self.best_solutions = None
        self.best_objectives = None

        #self.preprocessing = Normalizer()
        #self.preprocessing = MinMaxScaler()
        self.preprocessing = MaxAbsScaler()
    
    
    ## DVL execution for DTLZ benchmark problems
    def execute(self):

        upper = np.ones(self.problem.num_variables)
        lower = np.zeros(self.problem.num_variables)

        solutions = lhs(self.problem.num_variables, samples=self.samples)
        objectives = np.array([self.problem.evaluate(sol) for sol in solutions])
        
        if self.problem.num_objectives == 10:
            reference_points = np.asarray(generate_reference_points(self.problem.num_objectives, 3))
        else:
            reference_points = np.asarray(generate_reference_points(self.problem.num_objectives, self.problem.num_variables))
        
        ## The two strategies of using the inverse modeling by the default algorithm implementation:
        new_solutions = self.experimento1(objectives, solutions, reference_points, lower, upper, self.num_training)
        #new_solutions = self.experimento2(objectives, solutions, reference_points, lower, upper)    

        new_objectives = np.array([self.problem.evaluate(sol) for sol in new_solutions])
        
        solutions = np.concatenate((solutions, new_solutions))
        objectives = np.concatenate((objectives, new_objectives))

        self.best_solutions = new_solutions
        self.best_objectives = new_objectives

        best_hv = hvwfg.wfg(new_objectives, self.hv_ref)/self.normal
        print('Iter 1 hypervolume {}'.format(best_hv))
        prev_hv = best_hv
        for x in range(self.iterations - 1):

            new_solutions = self.experimento1(objectives, solutions, reference_points, lower, upper, self.num_training)   
            #new_solutions = self.experimento2(objectives, solutions, reference_points, lower, upper)

            new_objectives = np.array([self.problem.evaluate(sol) for sol in new_solutions])
            hv = hvwfg.wfg(new_objectives, self.hv_ref)/self.normal

            print('Iter {} hypervolume {}'.format(x+2, hv))

            solutions = np.concatenate((solutions, new_solutions))
            objectives = np.concatenate((objectives, new_objectives))

            if hv >= 0 and hv <= 1 and hv > best_hv:
                best_hv = hv
                self.best_solutions = new_solutions
                self.best_objectives = new_objectives            

            if best_hv != 0.0 and abs(hv-prev_hv) < self.diff:
                break
            else:
                prev_hv = hv

        print('Number of validations: {}'.format(objectives.shape[0]))
        print('Best hypervolume: {}'.format(best_hv))
        return best_hv

    def experimento1(self, objectives, solutions, reference_points, lower, upper, n):
        new_solutions = []
        num_sol = solutions.shape[0]
        num_var = solutions.shape[1]
        num_obj = objectives.shape[1]
        if num_sol < n:
            n = num_sol
        for rp in reference_points:
            dist = np.linalg.norm(objectives - rp, axis=1)

            #Criação de vetor auxiliar para ordenar pela distancia
            aux = np.zeros((num_sol,num_obj+1))
            aux[:,:-1] = objectives
            aux[:,num_obj] = dist

            obj_ordenado = aux[aux[:,num_obj].argsort()]

            nxt_objectives = np.zeros((n,num_obj))
            nxt_objectives = obj_ordenado[:n,:-1]

            #Criação de vetor auxiliar para ordenar pela distancia
            aux2 = np.zeros((num_sol,num_var+1))
            aux2[:,:-1] = solutions
            aux2[:,num_var] = dist

            sol_ordenado = aux2[aux2[:,num_var].argsort()]

            nxt_solutions = np.zeros((n,num_var))
            nxt_solutions = sol_ordenado[:n,:-1]
            rp_pred = np.zeros((1,nxt_solutions.shape[1]))
            if isinstance(self.pipeline, list):
                for j in range(nxt_solutions.shape[1]):
                    self.pipeline[j].fit(nxt_objectives, nxt_solutions[:,j])
                    rp_pred[:,j] = self.pipeline[j].predict([rp])

                rp_pred = rp_pred[0].tolist()
            else:
                self.pipeline.fit(nxt_objectives, nxt_solutions)
                rp_pred = self.pipeline.predict([rp])[0].tolist()
            new_solutions.append(truncate(rp_pred, lower, upper))

        return new_solutions

    ## Inverse modeling for the SUMO real problem:
    def experimento1Real(self, objectives_not, solutions, reference_points, lower, upper, n):
        
        objectives = self.preprocessing.fit_transform(objectives_not)
        new_solutions = []
        num_sol = solutions.shape[0]
        num_var = solutions.shape[1]
        num_obj = objectives.shape[1]
        if num_sol < n:
            n = num_sol
        for rp in reference_points:
            dist = np.linalg.norm(objectives - rp, axis=1)

            aux = np.zeros((num_sol,num_obj+1))
            aux[:,:-1] = objectives
            aux[:,num_obj] = dist

            obj_ordenado = aux[aux[:,num_obj].argsort()]

            nxt_objectives = np.zeros((n,num_obj))
            nxt_objectives = obj_ordenado[:n,:-1]

            aux2 = np.zeros((num_sol,num_var+1))
            aux2[:,:-1] = solutions
            aux2[:,num_var] = dist

            sol_ordenado = aux2[aux2[:,num_var].argsort()]

            nxt_solutions = np.zeros((n,num_var))
            nxt_solutions = sol_ordenado[:n,:-1]
            rp_pred = np.zeros((1,nxt_solutions.shape[1]))
            if isinstance(self.pipeline, list):
                for j in range(nxt_solutions.shape[1]):
                    self.pipeline[j].fit(nxt_objectives, nxt_solutions[:,j])
                    rp_pred[:,j] = self.pipeline[j].predict([rp])

                rp_pred = rp_pred[0].tolist()
            else:
                self.pipeline.fit(nxt_objectives, nxt_solutions)
                rp_pred = self.pipeline.predict([rp])[0].tolist()
            new_solutions.append(truncate([round(num) for num in rp_pred], lower, upper))

        return new_solutions

    def experimento2(self, objectives, solutions, reference_points, lower, upper):
        new_solutions = []
        self.pipeline.fit(objectives, solutions)

        for rp in reference_points:
            rp_pred = self.pipeline.predict([rp])[0].tolist()
            new_solutions.append(truncate(rp_pred, lower, upper))

        return new_solutions

    ## DVL execution for the SUMO real optimization problem
    def executeRealProblem(self):
        
        process_start = time.process_time()
        clock_start = time.time()

        upper = np.repeat(120, self.problem.num_variables)
        lower = np.repeat(20, self.problem.num_variables)

        solutions = (lhs(self.problem.num_variables, samples=self.samples) * 100) + 20
        solutions = solutions.astype(int)
        objectives = np.array([self.problem.evaluate(sol) for sol in solutions])
        

        print('Total Processing Time: {}'.format(self.num_process_time))
        process_wasted = time.process_time() - process_start
        print('LHS Wasted Time. {}'.format(process_wasted))


        if self.problem.num_objectives == 10 or self.problem.num_objectives == 6:
            reference_points = np.asarray(generate_reference_points(self.problem.num_objectives, 3))
        else:
            reference_points = np.asarray(generate_reference_points(self.problem.num_objectives, self.problem.num_variables))
        
        new_solutions = self.experimento1Real(objectives, solutions, reference_points, lower, upper, self.num_training)   
        new_array = [tuple(row) for row in new_solutions]
        new_solutions = np.unique(new_array, axis=0)

        new_objectives = np.array([self.problem.evaluate(sol) for sol in new_solutions])
        
        solutions = np.concatenate((solutions, new_solutions))
        objectives = np.concatenate((objectives, new_objectives))

        self.best_solutions = new_solutions
        self.best_objectives = new_objectives

        best_hv = hvwfg.wfg(new_objectives, self.hv_ref)/self.normal

        #print('Total Processing Time: {}'.format(self.num_process_time))
        process_wasted = time.process_time() - process_start
        print('Iter 1 hypervolume {}'.format(best_hv))
        print('Wasted Time. {}'.format(process_wasted))
        print('Number of validations: {}'.format(new_objectives.shape[0]))

        prev_hv = best_hv
        current_iteration = 1

        while process_wasted < self.num_process_time :
            new_solutions = self.experimento1Real(objectives, solutions, reference_points, lower, upper, self.num_training)   
            new_array = [tuple(row) for row in new_solutions]
            new_solutions = np.unique(new_array, axis=0)

            new_objectives = np.array([self.problem.evaluate(sol) for sol in new_solutions])
            hv = hvwfg.wfg(new_objectives, self.hv_ref)/self.normal

            process_wasted = time.process_time() - process_start
            current_iteration += 1
            print('Iter {} hypervolume {}'.format(current_iteration, hv))
            print('Wasted Time. {}'.format(process_wasted))
            print('Number of validations: {}'.format(new_objectives.shape[0]))

            solutions = np.concatenate((solutions, new_solutions))
            objectives = np.concatenate((objectives, new_objectives))

            if hv >= 0 and hv <= 1 and hv > best_hv:
                best_hv = hv
                self.best_solutions = new_solutions
                self.best_objectives = new_objectives            

            if best_hv != 0.0 and abs(hv-prev_hv) < self.diff:
                break
            else:
                prev_hv = hv


        process_final = time.process_time() - process_start
        clock_final = time.time() - clock_start
        print('Process Time {}'.format(process_final))    
        print('Clock Time {}'.format(clock_final))   
        with open('./Results/Experimento3/Bruto/time_'+str(self.num_process_time)+'_DVL_'+str(self.num_order)+'.txt', 'w') as f:
            print('Process Time {}'.format(process_final), file=f)    
            print('Clock Time {}'.format(clock_final), file=f)   

        # Saving DVL result into datasheet
        backup = np.concatenate((self.best_solutions, self.best_objectives), axis=1)
        var2 = pd.DataFrame(backup)
        var2.to_excel('./Results/Experimento3/Bruto/result_'+str(self.num_process_time)+'_DVL_'+str(self.num_order)+'.xlsx',header=False, index=False)

        print('Number of validations: {}'.format(objectives.shape[0]))
        print('Best hypervolume: {}'.format(best_hv))
        return best_hv
