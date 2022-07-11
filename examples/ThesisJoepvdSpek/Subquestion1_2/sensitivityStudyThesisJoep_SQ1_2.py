'''
This script runs the entire sensitivity study for
subquestion 1.2 of the MSc thesis of Joep van der Spek
(MSc Marine Technology TU Delft / Siemens Gamesa Renewable Energy)
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
import raft
import os
from datetime import datetime
from raft.sensitivityStudyFunctions import runRaftSensitivity, saveFigures


saveFiguresLocation = '/Volumes/GoogleDrive/My Drive/Graduation/Figures/Thesis/SQ1_2'
tic = time.perf_counter()
fileyaml = '/Users/joepvanderspek/PycharmProjects/RAFT/examples/ThesisJoepvdSpek/Subquestion1_2/SQ1_2.yaml'

# open the design YAML file and parse it into a dictionary for passing to raft

'''Change wind speed from 3 to 10.5'''
modelWS = runRaftSensitivity(fileyaml, variableSensitivitystudy = 'windSpeed', startValueSensitivityStudy = 3)

modelWS.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
modelWS.RMSmisalignresponse()
modelWS.plotPowerThrust()
modelWS.plotAeroTerms()
saveFigures(modelWS, saveFiguresLocation, identifier ='sq1_2')


'''Change wind heading from 0 to 120'''
modelWA = runRaftSensitivity(fileyaml, variableSensitivitystudy = 'windMisalignment', startValueSensitivityStudy = -120)

modelWA.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
modelWA.RMSmisalignresponse()
saveFigures(modelWA, saveFiguresLocation, identifier ='sq1_2')

'''Change Tp of Wave system 2 from 0.1 s to 15 s'''
modelWP2 = runRaftSensitivity(fileyaml, variableSensitivitystudy = 'wavePeriod2', startValueSensitivityStudy = 0.1)

modelWP2.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
modelWP2.RMSmisalignresponse()
saveFigures(modelWP2, saveFiguresLocation, identifier ='sq1_2')

'''Change misalignment angle Wave system 2 from 0 deg to 180 deg'''
modelMA1 = runRaftSensitivity(fileyaml, variableSensitivitystudy = 'misalignment', startValueSensitivityStudy = 0)

modelMA1.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
modelMA1.RMSmisalignresponse()
modelMA1.plotResponses_extended()

saveFigures(modelMA1, saveFiguresLocation, identifier ='sq1_2')

'''Change floater Orientation from 0 deg to 120 deg'''
modelFO = runRaftSensitivity(fileyaml, variableSensitivitystudy = 'floaterRotation', startValueSensitivityStudy = 0)

modelFO.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
modelFO.RMSmisalignresponse(twoDOF=True, RootRMS=False)
modelFO.plotResponses_extended()

saveFigures(modelFO, saveFiguresLocation, identifier ='sq1_2')

toc = time.perf_counter()
totaltime = toc - tic
print(f"Ran total sensitivity study in {totaltime:0.2f} seconds, {round(totaltime/60)} minutes")

plt.show()

