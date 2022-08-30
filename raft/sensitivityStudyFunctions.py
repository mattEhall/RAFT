'''
Version 1.0 set-up by Joep van der Spek (joep.spek@siemensgamesa.com)
Available for open-source use.
'''

import raft
import os
import yaml
from datetime import datetime
import matplotlib.pyplot as plt

def runRaftSensitivity(yamlDesign, variableSensitivitystudy, startValueSensitivityStudy):
    '''This function returns an analyzed model object (raft_model.py) for either a parametric study or a design yaml.

     yamlDesign
        yamlDesign, either a string with its location or a design dict
     variableSensitivitystudy
        variable selector for performing a sensitivity study
     startValueSensitivityStudy
        Start value of sensitivity study, replaces the value in original 'cases' part of design dict for that specific value

     '''
    if isinstance(yamlDesign, str):
        with open(yamlDesign , 'r') as file:
            design = yaml.safe_load(file)
    else:
        design = yamlDesign

    modelObject = raft.Model(design, parametricAnalysisBool=True, variableSensitivitystudy=variableSensitivitystudy,
                         startValueSensitivityStudy=startValueSensitivityStudy)
    modelObject.analyzeUnloaded()
    modelObject.solveEigen()
    modelObject.analyzeCases(display=False, numberOfCores=-2)

    return modelObject

def saveFigures(modelObject, location, identifier = 'base'):
    ''' Helper function to save all figures at a specific location

    modelObject
        raft_model.py object that includes the relevant 'changeType' as an identifier for the string
    location
        folder location on where to store results
    identifier
        string with identifier to make folder and figure titles mode readable.


    '''
    datetimeobj = datetime.now()
    folder = datetimeobj.strftime("%d_%b_%Y_%H_%M")
    os.mkdir(f'{location}/{folder}_{modelObject.changeType}')
    for i in plt.get_fignums():
        plt.figure(i).savefig(f'{location}/{folder}_{modelObject.changeType}/{identifier}_{modelObject.changeType}_{i}.pdf')
        plt.close(i)
    return print(f'Plotting completed of {modelObject.changeType}')