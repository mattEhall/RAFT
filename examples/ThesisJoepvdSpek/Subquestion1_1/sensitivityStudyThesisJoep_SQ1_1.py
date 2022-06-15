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


'''Change floater Orientation from 0 deg to 120 deg'''
modelFO = runRaftSensitivity(file, variableSensitivitystudy = 'floaterRotation', startValueSensitivityStudy = 0)

modelFO.plotBEMTerms()
modelFO.plotAeroTerms()
modelFO.plotCouplingTerms()
modelFO.plotCouplingContribution()
modelFO.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)

saveFigures(modelFO,saveFiguresLocation)

'''Change misalignment angle Wave system 2 from 0 deg to 180 deg'''
modelMA1 = runRaftSensitivity(file, variableSensitivitystudy = 'misalignment', startValueSensitivityStudy = 0)

modelMA1.plotBEMTerms()
modelMA1.plotAeroTerms()
modelMA1.plotCouplingTerms()
modelMA1.plotCouplingContribution()
modelMA1.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)

saveFigures(modelMA1, saveFiguresLocation)

'''Change wave period of system 2 from 0.1 s to 25 seconds'''
modelWP2 = runRaftSensitivity(file, variableSensitivitystudy = 'wavePeriod2', startValueSensitivityStudy = 0.1)

modelWP2.plotBEMTerms()
modelWP2.plotAeroTerms()
modelWP2.plotCouplingTerms()
modelWP2.plotCouplingContribution()
modelWP2.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
modelWP2.RMSmisalignresponse(twoDOF=False, RootRMS= True)

saveFigures(modelWP2,saveFiguresLocation)

'''Change Wave Height of system 2 from 0 m to 6 m'''
modelWH2 = runRaftSensitivity(file, variableSensitivitystudy = 'waveHeight2', startValueSensitivityStudy = 0)

modelWH2.plotBEMTerms()
modelWH2.plotAeroTerms()
modelWH2.plotCouplingTerms()
modelWH2.plotCouplingContribution()
modelWH2.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
modelWH2.RMSmisalignresponse(twoDOF=False, RootRMS= True)

saveFigures(modelWH2,saveFiguresLocation)


modelFO.plot(hideGrid=False)

toc = time.perf_counter()
totaltime = toc - tic
print(f"Ran total sensitivity study in {totaltime:0.2f} seconds, {round(totaltime/60)} minutes")

plt.show()

