'''
This script runs the entire sensitivity study for
subquestion 1.1 of the MSc thesis of Joep van der Spek
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

saveFiguresLocation = '/Volumes/GoogleDrive/My Drive/Graduation/Figures/Thesis/SQ1_1'
tic = time.perf_counter()
# open the design YAML file and parse it into a dictionary for passing to raft
file = '/Users/joepvanderspek/PycharmProjects/RAFT/examples/ThesisJoepvdSpek/Subquestion1_1/SQ1_1.yaml'



# '''Change wind angle from -120 to 120'''
# modelWA = runRaftSensitivity(file, variableSensitivitystudy = 'windMisalignment', startValueSensitivityStudy = -120)
#
# modelWA.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
# modelWA.RMSmisalignresponse()
# modelWA.plotCouplingContribution()
# modelWA.plotCouplingContribution(diagonal = False)
#
# saveFigures(modelWA, saveFiguresLocation, identifier ='sq1_1')
#
# '''Change wind speed from 3 to 10.5'''
# modelWS = runRaftSensitivity(file, variableSensitivitystudy = 'windSpeed', startValueSensitivityStudy = 3)
#
# modelWS.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
# modelWS.RMSmisalignresponse()
# modelWS.plotPowerThrust()
# modelWS.plotAeroTerms()
# modelWS.plotCouplingTerms()
# modelWS.plotCouplingTerms(diagonal=False)
# modelWS.plotCouplingContribution()
# modelWS.plotCouplingContribution(diagonal = False)
#
# saveFigures(modelWS, saveFiguresLocation, identifier ='sq1_1')
#
# '''Change floater Orientation from 0 deg to 120 deg'''
# modelFO = runRaftSensitivity(file, variableSensitivitystudy = 'floaterRotation', startValueSensitivityStudy = 0)
#
# modelFO.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
# modelFO.RMSmisalignresponse()
# modelFO.plotAeroTerms()
# modelFO.plotCouplingTerms()
# modelFO.plotCouplingTerms(diagonal=False)
# modelFO.plotCouplingContribution()
# modelFO.plotCouplingContribution(diagonal = False)
#
# saveFigures(modelFO,saveFiguresLocation, identifier ='sq1_1')
#
# '''Change misalignment angle Wave system 2 from 0 deg to 180 deg'''
# modelMA1 = runRaftSensitivity(file, variableSensitivitystudy = 'misalignment', startValueSensitivityStudy = 0)
#
# modelMA1.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
# modelMA1.RMSmisalignresponse(twoDOF=False, RootRMS= True)
# modelMA1.plotCouplingTerms()
# modelMA1.plotCouplingTerms(diagonal=False)
# modelMA1.plotCouplingContribution()
# modelMA1.plotCouplingContribution(diagonal = False)
#
# saveFigures(modelMA1, saveFiguresLocation, identifier ='sq1_1')

'''Change wave period of system 2 from 1 s to 30 seconds'''
modelWP2 = runRaftSensitivity(file, variableSensitivitystudy = 'wavePeriod2', startValueSensitivityStudy = 1)

modelWP2.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
modelWP2.RMSmisalignresponse(twoDOF=False, RootRMS= True)
modelWP2.plotCouplingContribution()
modelWP2.plotCouplingContribution(diagonal = False)

saveFigures(modelWP2,saveFiguresLocation, identifier ='sq1_1')



# modelFO.plot(hideGrid=False)

toc = time.perf_counter()
totaltime = toc - tic
print(f"Ran total sensitivity study in {totaltime:0.2f} seconds, {round(totaltime/60)} minutes")

plt.show()

