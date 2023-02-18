from optproblems import dtlz
from optproblems import Individual
import numpy as np
import pandas as pd

class ProblemsUtil():

    def __init__(self,
                problem=None,
                objectives=3,
                variables=None):

        if problem == None:
            raise Exception("Cannot initializate without a problem.")

        self.problem_label = problem
        if self.problem_label == 'DTLZ1':  
            self.updateProblem(variables, objectives, 5)
            self.problem = dtlz.DTLZ1(self.num_objectives,self.num_variables)
        elif self.problem_label == 'DTLZ2':
            self.updateProblem(variables, objectives, 10)
            self.problem = dtlz.DTLZ2(self.num_objectives,self.num_variables)
        elif self.problem_label == 'DTLZ3':
            self.updateProblem(variables, objectives, 10)
            self.problem = dtlz.DTLZ3(self.num_objectives,self.num_variables)
        elif self.problem_label == 'DTLZ4':
            self.updateProblem(variables, objectives, 10)
            self.problem = dtlz.DTLZ4(self.num_objectives,self.num_variables)
        elif self.problem_label == 'DTLZ5':
            self.updateProblem(variables, objectives, 10)
            self.problem = dtlz.DTLZ5(self.num_objectives,self.num_variables)
        elif self.problem_label == 'DTLZ6':
            self.updateProblem(variables, objectives, 10)
            self.problem = dtlz.DTLZ6(self.num_objectives,self.num_variables)
        elif self.problem_label == 'DTLZ7':
            self.updateProblem(variables, objectives, 20)
            self.problem = dtlz.DTLZ7(self.num_objectives,self.num_variables)
    
    def updateProblem(self, variables, objectives, k):
        if variables == None:          
            self.num_variables = k + objectives - 1 
        else:
            self.num_variables = variables
        self.num_objectives = objectives

    def evaluate(self,
                    solutions=None):
        return self.problem.objective_function(solutions)

    def optimal(self,
                    samples=91):
        
        if self.problem_label == 'DTLZ1' or self.problem_label == 'DTLZ2' or self.problem_label == 'DTLZ3' or self.problem_label == 'DTLZ4' or self.problem_label == 'DTLZ5' or self.problem_label == 'DTLZ6' or self.problem_label == 'DTLZ7':
            pd_solutions = pd.read_csv('./Front/'+self.problem_label+'_'+str(self.num_objectives)+'_12_variaveis.csv', header=0 , sep=',', index_col=0) 
            solutions = pd_solutions.to_numpy()
            optimal_sol = solutions.copy(order='C')
            optimal_sol = optimal_sol[:samples]

            pd_objectives = pd.read_csv('./Front/'+self.problem_label+'_'+str(self.num_objectives)+'_pareto.csv', header=0 , sep=',', index_col=0) 
            objectives = pd_objectives.to_numpy()
            objectives = objectives.copy(order='C')
            objectives = objectives[:samples]
        else:
            solutions = self.problem.get_optimal_solutions(samples)
            objectives = np.array([self.problem.objective_function(sol.phenome) for sol in solutions])
            optimal_sol = np.array([sol.phenome for sol in solutions])

        return optimal_sol, objectives

    def referenceHV(self):
        if self.problem_label == 'DTLZ1':   
            ## DVL Optimization Experiment
            return np.repeat(300.0, self.num_objectives)

            ## DVL Framework Optimization Experiment
            #if self.num_objectives == 3:
            #    return np.repeat(1.0, self.num_objectives)
            #else:
            #   return np.repeat(5.0, self.num_objectives)
        elif self.problem_label == 'DTLZ2':
            ## DVL Optimization Experiment
            return np.repeat(2.0, self.num_objectives)

            ## DVL Framework Optimization Experiment
            #if self.num_objectives == 3:
            #    return np.repeat(2.0, self.num_objectives)
            #else:
            #    return np.repeat(1.5, self.num_objectives)
        elif self.problem_label == 'DTLZ3':
            ## DVL Optimization Experiment
            #return np.repeat(700.0, self.num_objectives)

            ## DVL Framework Optimization Experiment
            if self.num_objectives == 3:
                return np.repeat(10.0, self.num_objectives) 
            else:
                return np.repeat(50.0, self.num_objectives) 
        elif self.problem_label == 'DTLZ4':
            ## DVL Optimization Experiment
            return np.repeat(2.0, self.num_objectives)
            
            ## DVL Framework Optimization Experiment
            #if self.num_objectives == 3:
            #    return np.repeat(2.0, self.num_objectives)
            #else:
            #    return np.repeat(1.5, self.num_objectives)
        elif self.problem_label == 'DTLZ5':
            ## DVL Optimization Experiment
            result = np.zeros(self.num_objectives)
            for i in range(self.num_objectives):
                result[i] = 2 + (2 * (i + 1))
            return result

            ## DVL Framework Optimization Experiment
            #if self.num_objectives == 3:
            #    result = np.zeros(self.num_objectives)
            #    for i in range(self.num_objectives):
            #        result[i] = 3 + (2 * i)
            #    return result
            #else:
            #    return np.repeat(4.0, self.num_objectives)
        elif self.problem_label == 'DTLZ6':
            ## DVL Optimization Experiment
            result = np.zeros(self.num_objectives)
            for i in range(self.num_objectives):
                result[i] = 2 + (8 * (i + 1))
            return result

            ## DVL Framework Optimization Experiment
            #result = np.zeros(self.num_objectives)
            #if self.num_objectives == 3:
            #    for i in range(self.num_objectives):
            #        result[i] = 2 + (4 * (i + 1))
            #    
            #else:
            #    for i in range(self.num_objectives):
            #        result[i] = 2 + i
            #return result           
        elif self.problem_label == 'DTLZ7':
            ## DVL and DVL Framework Optimization Experiments
            result = np.zeros(self.num_objectives)
            for i in range(self.num_objectives):
                result[i] = 2 + (8 * (i + 1))
                #result[i] = 2 + (4 * (i + 1))
            return result