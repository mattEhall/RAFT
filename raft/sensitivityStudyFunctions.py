import raft
import os
import yaml
from datetime import datetime
import matplotlib.pyplot as plt

def runRaftSensitivity(yamlDesign, variableSensitivitystudy, startValueSensitivityStudy):

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

def saveFigures(modelObject, location, identifier = 'question1'):
    datetimeobj = datetime.now()
    folder = datetimeobj.strftime("%d_%b_%Y_%H_%M")
    os.mkdir(f'{location}/{folder}_{modelObject.changeType}')
    for i in plt.get_fignums():
        plt.figure(i).savefig(f'{location}/{folder}_{modelObject.changeType}/{identifier}_{modelObject.changeType}_{i}.pdf')
        plt.close(i)
    return print(f'Plotting completed of {modelObject.changeType}')