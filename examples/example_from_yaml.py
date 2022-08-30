# example script for running RAFT from a YAML input file
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
import raft
import os
from datetime import datetime

tic = time.perf_counter()
# open the design YAML file and parse it into a dictionary for passing to raft
with open('VolturnUS-S_example.yaml') as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

# Create the RAFT model (will set up all model objects based on the design dict)
model = raft.Model(design, parametricAnalysisBool=False, variableSensitivitystudy='windSpeed', startValueSensitivityStudy = 4.1)
# model = raft.Model(design)
# Evaluate the system properties and equilibrium position before loads are applied
model.analyzeUnloaded()

# Compute natural frequencie
model.solveEigen()

# Simule the different load cases
model.analyzeCases(display=True, numberOfCores= -2)

# Plot the power spectral densities from the load cases
# model.plotResponses_extended()
# model.plotBEMTerms()
# model.plotBEMTerms(diagonal=False)
# model.plotAeroTerms()
# model.plotCouplingTerms()
# model.plotCouplingTerms(diagonal=False)
# model.plotTowerBaseResponse(plot='polar', plot_eq_stress_angles=True, include_surface = False)
# model.RMSmisalignresponse(RootRMS= False)
# model.plotPowerThrust()
# model.plotCouplingContribution()
#
# model.plotCouplingContribution(diagonal = False)
# model.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=False)

# Visualize the system in its most recently evaluated mean offset position
model.plot(hideGrid=False)

print(f'M_struc_tot = {model.fowtList[0].M_struc}')
print(f'M tower = {model.fowtList[0].mtower}')
print(f'M_struc_sub_tot = {model.fowtList[0].M_struc_subCM}')


# print(model.results)
# print(model.Xi)
toc = time.perf_counter()
print(f"Ran all cases in {toc - tic:0.4f} seconds")
# datetimeobj = datetime.now()
# folder = datetimeobj.strftime("%d_%b_%Y_%H_%M")
# os.mkdir(f'/Volumes/GoogleDrive/My Drive/Graduation/Figures/Thesis/Base_Case/{folder}_{model.changeType}')
# for i in plt.get_fignums():
#     plt.figure(i).savefig(f'/Volumes/GoogleDrive/My Drive/Graduation/Figures/Thesis/Base_Case/{folder}_{model.changeType}/Base_Case_{i}.pdf')

plt.show()

