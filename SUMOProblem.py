import os  
import subprocess
import sys
import shutil
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', "tools"))
sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
    os.path.dirname(__file__), "..", "..", "..")), "tools"))
from sumolib import checkBinary  # noqa
import xml.etree.ElementTree as ET
import numpy as np
import time


class SUMOProblem():

    def __init__(self,
                scenario=1,
                config_file='config.sumocfg',
                output_file='tripinfos.xml',
                add_path='./SUMO/add_default.xml'):

        if scenario == 1:
            self.scenario_path = './SUMO/cenario1'
            self.scenario_logics = ['2']
            self.scenario_phases = ['GGGrrrGGGrrr', 'yyyrrryyyrrr', 'rrrGGGrrrGGG', 'rrryyyrrryyy']
        elif scenario == 2:
            self.scenario_path = './SUMO/cenario2'
            self.scenario_logics = ['1', '2', '3', '9']
            self.scenario_phases = ['GGGrrrGGGrrr', 'yyyrrryyyrrr', 'rrrGGGrrrGGG', 'rrryyyrrryyy']
        elif scenario == 3:
            self.scenario_path = './SUMO/cenario3'
            self.scenario_logics = ['10', '11', '14', '15', '16', '4', '5', '6', '9']
            self.scenario_phases = ['GGrrGGrr', 'yyrryyrr', 'rrGGrrGG', 'rryyrryy']
        else:
            raise Exception("Scenario doesn't exist!")    
        if not os.path.isdir(self.scenario_path):  
            raise Exception("Scenario path doesn't exist!")      

        self.config_file = config_file
        self.config_path = self.scenario_path+'/'+self.config_file
        if not os.path.isfile(self.config_path):  
            raise Exception("Config path file doesn't exist.")      

        self.output_file = output_file
        self.output_path = self.scenario_path+'/output/'+self.output_file
        self.add_path = add_path
        self.num_objectives = 6
        self.num_variables = 8
    
    def evaluate(self,
                    solutions=None):
        # transforming inputs
        self.build_input_xml(solutions)
        # running simulation
        sumoBinary = checkBinary('sumo')                    
        retcode = subprocess.call(
            [sumoBinary, "-c", self.config_path, "--no-step-log", "--no-warnings"], stdout=open(os.devnull, "w"), stderr=sys.stderr)
        #stdout=sys.stdout,
        #stdout=open(os.devnull, "w")
        #print(">> Simulation closed with status %s" % retcode)
        #sys.stdout.flush()
        # transforming outputs
        return self.parse_objectives()

    def build_input_xml(self, solutions=None):
        tree = ET.parse(self.add_path)
        root = tree.getroot()   
        
        for i,logic in enumerate(self.scenario_logics):
            tlLogic = ET.SubElement(root, 'tlLogic')
            tlLogic.set('id', logic)
            tlLogic.set('type', 'static')
            tlLogic.set('programID', '1')
            tlLogic.set('offset', '0')

            phase1 = ET.SubElement(tlLogic, 'phase')
            phase1.set('duration', str(solutions[i*2]))
            phase1.set('state', self.scenario_phases[0])
            
            phase2 = ET.SubElement(tlLogic, 'phase')
            phase2.set('duration', '4')
            phase2.set('state', self.scenario_phases[1])
            
            phase3 = ET.SubElement(tlLogic, 'phase')
            phase3.set('duration', str(solutions[(i*2)+1]))
            phase3.set('state', self.scenario_phases[2])
            
            phase4 = ET.SubElement(tlLogic, 'phase')
            phase4.set('duration', '4')
            phase4.set('state', self.scenario_phases[3])

        tree.write(self.scenario_path+'/'+'add.xml')

    def parse_objectives(self):
        tree = ET.parse(self.output_path)
        root = tree.getroot()     
        objectives = np.zeros(self.num_objectives)   
        for tripinfo in root:
            objectives[0] += float(tripinfo.get('departDelay'))
            objectives[1] += float(tripinfo.get('duration'))
            objectives[2] += float(tripinfo.get('waitingTime'))            
            objectives[3] += float(tripinfo.get('timeLoss'))
            for emission in tripinfo:      
                objectives[4] += float(emission.get('CO2_abs'))/1000 
                objectives[5] += float(emission.get('fuel_abs'))

        return objectives

    def referenceHV(self):
        return np.array([17499854.51, 2397784, 1816123, 2287134.57000001, 11522710.3071638, 4953205.11018303])

#problem = SUMOProblem(scenario=2)
#process_start = time.process_time()
#clock_start = time.time()
#objectives = problem.evaluate([11, 20, 20, 13, 20, 20, 120, 20, 20, 40, 40, 20, 20, 20, 40, 40, 20, 20])
#print('Process Time {}'.format(time.process_time() - process_start))    
#print('Clock Time {}'.format(time.time() - clock_start))    
#print('Objectives {}'.format(objectives))    


