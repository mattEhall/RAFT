# example script for running RAFT from a YAML input file
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
import raft

tic = time.perf_counter()
# open the design YAML file and parse it into a dictionary for passing to raft
with open('VolturnUS-S_example.yaml') as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

# Create the RAFT model (will set up all model objects based on the design dict)
model = raft.Model(design)  

# Evaluate the system properties and equilibrium position before loads are applied
model.analyzeUnloaded()

# Compute natural frequencie
model.solveEigen()

# Simule the different load cases
model.analyzeCases(display=1)

# Plot the power spectral densities from the load cases
model.plotResponses_extended()

model.plotTowerBaseResponse(include_surface = False)
model.plotTowerBaseResponse(plot = 'polar', plot_eq_stress_angles=True)
model.RMSmisalignresponse()
# model.plotCouplingTerms()
# model.plotBEMTerms()
# model.plotCouplingContribution()
# Visualize the system in its most recently evaluated mean offset position
model.plot(hideGrid=False)

# print(model.results)
# print(model.Xi)
toc = time.perf_counter()
print(f"Ran all cases in {toc - tic:0.4f} seconds")

plt.show()
for i in plt.get_fignums():
    plt.figure(i).savefig(f'figures/figure22042022_{i}.png')
