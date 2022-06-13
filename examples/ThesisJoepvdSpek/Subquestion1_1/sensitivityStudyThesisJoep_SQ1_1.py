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

'''Change floater Orientation from 0 deg to 120 deg'''
modelFO = runRaftSensitivity('SQ1_1.yaml', variableSensitivitystudy = 'floaterRotation', startValueSensitivityStudy = 0)

modelFO.plotBEMTerms()
modelFO.plotAeroTerms()
modelFO.plotCouplingTerms()
modelFO.plotCouplingContribution()
modelFO.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)

saveFigures(modelFO,saveFiguresLocation)

'''Change wave period of system 2 from 0.1 s to 25 seconds'''
modelWP2 = runRaftSensitivity('SQ1_1.yaml', variableSensitivitystudy = 'wavePeriod2', startValueSensitivityStudy = 0.1)

modelWP2.plotBEMTerms()
modelWP2.plotAeroTerms()
modelWP2.plotCouplingTerms()
modelWP2.plotCouplingContribution()
modelWP2.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)

saveFigures(modelFO,saveFiguresLocation)

# Simulate the different load cases
# model.analyzeCases(display=False, numberOfCores= -2)

# Plot the power spectral densities from the load cases
# model.plotResponses_extended()
# model.plotBEMTerms()
# model.plotAeroTerms()
# model.plotCouplingTerms()
# model.plotTowerBaseResponse(include_surface = False)
# model.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
# model.RMSmisalignresponse(RootRMS= False)
# model.plotPowerThrust()
# model.plotCouplingTerms()
# model.plotCouplingContribution()
# Visualize the system in its most recently evaluated mean offset position
modelFO.plot(hideGrid=False)

toc = time.perf_counter()
totaltime = toc - tic
print(f"Ran total sensitivity study in {totaltime:0.2f} seconds, {round(totaltime/60)} minutes")

plt.show()

