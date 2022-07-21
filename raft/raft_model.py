# RAFT's main model class

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, rc
import yaml
import matplotlib as mpl
from cycler import cycler

monochrome = (cycler('color', ['k', '0.75']) *
              cycler('linestyle', ['-', '--', ':']))
plt.rc('axes', prop_cycle=monochrome)


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
rc('legend',fontsize=4)
rc('font', size=6)
plt.rcParams['axes.formatter.limits'] = (-3, 3)
# rc('text', usetex=True)

from fatiguepy import *

from joblib import Parallel, delayed
from tqdm import tqdm

import moorpy as mp
import raft.raft_fowt as fowt
from raft.helpers import *
import gc

# import F6T1RNA as structural    # import turbine structural model functions

raft_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TwoPi = 2.0 * np.pi


class Model():

    def __init__(self, design, nTurbines=1, parametricAnalysisBool=False, variableSensitivitystudy=None,
                 startValueSensitivityStudy=None):
        '''
        Empty frequency domain model initialization function

        design : dict
            Dictionary of all the design info from turbine to platform to moorings
        nTurbines
            could in future be used to set up any number of identical turbines
        '''

        self.fowtList = []
        self.coords = []

        self.nDOF = 0  # number of DOFs in system

        if parametricAnalysisBool and variableSensitivitystudy is not None:
            self.changeType = variableSensitivitystudy
            print(f'Sensitivity study for variable {self.changeType} starting now...')
            self.design = parametricAnalysisBuilder(design, self.changeType, startValueSensitivityStudy,
                                             parametricAnalysisBool)
        else:

            self.design = design  # save design dictionary for possible later use/reference
            self.changeType = ""
        # print(self.design['cases'])
        # parse settings
        if not 'settings' in design:  # if settings field not in input data
            design['settings'] = {}  # make an empty one to avoid errors

        min_freq = getFromDict(design['settings'], 'min_freq', default=0.01,
                               dtype=float)  # [Hz] lowest frequency to consider, also the frequency bin width
        max_freq = getFromDict(design['settings'], 'max_freq', default=1.00,
                               dtype=float)  # [Hz] highest frequency to consider
        self.XiStart = getFromDict(design['settings'], 'XiStart', default=0.1,
                                   dtype=float)  # sets initial amplitude of each DOF for all frequencies
        self.nIter = getFromDict(design['settings'], 'nIter', default=15,
                                 dtype=int)  # sets how many iterations to perform in Model.solveDynamics()

        self.w = np.arange(min_freq, max_freq + 0.5 * min_freq,
                           min_freq) * 2 * np.pi  # angular frequencies to analyze (rad/s)
        self.nw = len(self.w)  # number of frequencies
        self.numAngles = int(getFromDict(design['settings'], 'numAngles',
                                         default=73))  # to cover 0 to 360 in steps of 5 for calculation of stress around tb
        self.anglesArray = np.linspace(0, 2 * np.pi, self.numAngles)
        self.printStatements = getFromDict(design['settings'], 'printStatements', default=True)
        self.latex_width = getFromDict(design['settings'], 'width_latex', default = 400, dtype=float)
        # process mooring information 
        self.ms = mp.System()
        self.ms.parseYAML(design['mooring'])

        # depth and wave number        
        self.depth = getFromDict(design['site'], 'water_depth', dtype=float)
        self.k = np.zeros(self.nw)  # wave number
        for i in range(self.nw):
            self.k[i] = waveNumber(self.w[i], self.depth)

        # set up the FOWT here  <<< only set for 1 FOWT for now <<<
        self.fowtList.append(fowt.FOWT(design, self.w, self.ms.bodyList[0], depth=self.depth))
        self.coords.append([0.0, 0.0])
        self.nDOF += 6

        self.ms.bodyList[0].type = -1  # need to make sure it's set to a coupled type

        try:
            self.ms.initialize()  # reinitialize the mooring system to ensure all things are tallied properly etc.
        except Exception as e:
            raise RuntimeError('An error occured when initializing the mooring system: ' + e.message)

        self.results = {}  # dictionary to hold all results from the model

    def addFOWT(self, fowt, xy0=[0, 0]):
        '''(not used currently) Adds an already set up FOWT to the frequency domain model solver.'''

        self.fowtList.append(fowt)
        self.coords.append(xy0)
        self.nDOF += 6

        # would potentially need to add a mooring system body for it too <<<

    """
    def setEnv(self, Hs=8, Tp=12, spectrum='unit', V=10, beta=0, Fthrust=0):

        self.env = Env()
        self.env.Hs       = Hs
        self.env.Tp       = Tp
        self.env.spectrum = spectrum
        self.env.V        = V
        self.env.beta     = beta
        self.Fthrust      = Fthrust

        for fowt in self.fowtList:
            fowt.setEnv(Hs=Hs, Tp=Tp, V=V, spectrum=spectrum, beta=beta, Fthrust=Fthrust)
    """

    def analyzeUnloaded(self):
        '''This calculates the system properties under undloaded coonditions: equilibrium positions, natural frequencies, etc.'''

        # calculate the system's constant properties
        # self.calcSystemConstantProps()
        for fowt in self.fowtList:
            fowt.calcStatics()
            # fowt.calcBEM()
            fowt.calcHydroConstants(dict(wave_spectrum='still', wave_heading=0, add_waveheading=False))

        # get mooring system characteristics about undisplaced platform position (useful for baseline and verification)
        try:
            self.C_moor0 = self.ms.getCoupledStiffness(
                lines_only=True)  # this method accounts for eqiuilibrium of free objects in the system
            self.F_moor0 = self.ms.getForces(DOFtype="coupled", lines_only=True)
        except Exception as e:
            raise RuntimeError(
                'An error occured when getting linearized mooring properties in undisplaced state: ' + e.message)

        self.results['properties'] = {}  # signal this data is available by adding a section to the results dictionary

        # calculate platform offsets and mooring system equilibrium state
        self.calcMooringAndOffsets()
        self.results['properties']['offset_unloaded'] = self.fowtList[0].Xi0

        # TODO: add printing of summary info here - mass, stiffnesses, etc

    def analyzeCases(self, display=False, numberOfCores=1):
        '''This runs through all the specified load cases, building a dictionary of results.'''

        nCases = len(self.design['cases']['data'])
        nLines = len(self.ms.lineList)

        # set up output arrays for load cases

        self.results['case_metrics'] = {}
        self.results['case_metrics']['surge_avg'] = np.zeros(nCases)
        self.results['case_metrics']['surge_std'] = np.zeros(nCases)
        self.results['case_metrics']['surge_max'] = np.zeros(nCases)
        self.results['case_metrics']['surge_PSD'] = np.zeros(
            [nCases, self.nw])  # adding PSDs as well. Could put behind an if statement if this slows things down

        self.results['case_metrics']['sway_avg'] = np.zeros(nCases)
        self.results['case_metrics']['sway_std'] = np.zeros(nCases)
        self.results['case_metrics']['sway_max'] = np.zeros(nCases)
        self.results['case_metrics']['sway_PSD'] = np.zeros([nCases, self.nw])

        self.results['case_metrics']['heave_avg'] = np.zeros(nCases)
        self.results['case_metrics']['heave_std'] = np.zeros(nCases)
        self.results['case_metrics']['heave_max'] = np.zeros(nCases)
        self.results['case_metrics']['heave_PSD'] = np.zeros([nCases, self.nw])

        self.results['case_metrics']['roll_avg'] = np.zeros(nCases)
        self.results['case_metrics']['roll_std'] = np.zeros(nCases)
        self.results['case_metrics']['roll_max'] = np.zeros(nCases)
        self.results['case_metrics']['roll_PSD'] = np.zeros([nCases, self.nw])

        self.results['case_metrics']['pitch_avg'] = np.zeros(nCases)
        self.results['case_metrics']['pitch_std'] = np.zeros(nCases)
        self.results['case_metrics']['pitch_max'] = np.zeros(nCases)
        self.results['case_metrics']['pitch_PSD'] = np.zeros([nCases, self.nw])

        self.results['case_metrics']['yaw_avg'] = np.zeros(nCases)
        self.results['case_metrics']['yaw_std'] = np.zeros(nCases)
        self.results['case_metrics']['yaw_max'] = np.zeros(nCases)
        self.results['case_metrics']['yaw_PSD'] = np.zeros([nCases, self.nw])

        # nacelle acceleration
        self.results['case_metrics']['AxRNA_avg'] = np.zeros(nCases)
        self.results['case_metrics']['AxRNA_std'] = np.zeros(nCases)
        self.results['case_metrics']['AxRNA_max'] = np.zeros(nCases)
        self.results['case_metrics']['AxRNA_PSD'] = np.zeros([nCases, self.nw])
        # tower base bending moment FA
        self.results['case_metrics']['Mbase_avg'] = np.zeros(nCases)
        self.results['case_metrics']['Mbase_std'] = np.zeros(nCases)
        self.results['case_metrics']['Mbase_max'] = np.zeros(nCases)
        self.results['case_metrics']['Mbase_PSD'] = np.zeros([nCases, self.nw])
        self.results['case_metrics']['Mbase_sig'] = np.zeros([nCases, self.nw], dtype=complex)
        self.results['case_metrics']['Mbase_DEL'] = np.zeros(nCases)
        # tower base bending moment SS
        self.results['case_metrics']['MbaseSS_avg'] = np.zeros(nCases)
        self.results['case_metrics']['MbaseSS_std'] = np.zeros(nCases)
        self.results['case_metrics']['MbaseSS_max'] = np.zeros(nCases)
        self.results['case_metrics']['MbaseSS_PSD'] = np.zeros([nCases, self.nw])
        self.results['case_metrics']['MbaseSS_sig'] = np.zeros([nCases, self.nw], dtype=complex)
        self.results['case_metrics']['MbaseSS_DEL'] = np.zeros(nCases)
        # rotor speed
        self.results['case_metrics']['omega_avg'] = np.zeros(nCases)
        self.results['case_metrics']['omega_std'] = np.zeros(nCases)
        self.results['case_metrics']['omega_max'] = np.zeros(nCases)
        self.results['case_metrics']['omega_PSD'] = np.zeros([nCases, self.nw])
        # generator torque
        self.results['case_metrics']['torque_avg'] = np.zeros(nCases)
        self.results['case_metrics']['torque_std'] = np.zeros(nCases)
        self.results['case_metrics']['torque_max'] = np.zeros(nCases)
        self.results['case_metrics']['torque_PSD'] = np.zeros([nCases, self.nw])
        # rotor power 
        self.results['case_metrics']['power_avg'] = np.zeros(nCases)
        self.results['case_metrics']['power_std'] = np.zeros(nCases)
        self.results['case_metrics']['power_max'] = np.zeros(nCases)
        self.results['case_metrics']['power_PSD'] = np.zeros([nCases, self.nw])
        # rotor thrust
        self.results['case_metrics']['thrust_avg'] = np.zeros(nCases)
        # collective blade pitch
        self.results['case_metrics']['bPitch_avg'] = np.zeros(nCases)
        self.results['case_metrics']['bPitch_std'] = np.zeros(nCases)
        self.results['case_metrics']['bPitch_max'] = np.zeros(nCases)
        self.results['case_metrics']['bPitch_PSD'] = np.zeros([nCases, self.nw])
        # mooring tension
        self.results['case_metrics']['Tmoor_avg'] = np.zeros(
            [nCases, 2 * nLines])  # 2d array, for each line in each case?
        self.results['case_metrics']['Tmoor_std'] = np.zeros([nCases, 2 * nLines])
        self.results['case_metrics']['Tmoor_max'] = np.zeros([nCases, 2 * nLines])
        self.results['case_metrics']['Tmoor_DEL'] = np.zeros([nCases, 2 * nLines])
        self.results['case_metrics']['Tmoor_PSD'] = np.zeros([nCases, 2 * nLines, self.nw])

        # wind and wave spectra for reference
        self.results['case_metrics']['wind_PSD'] = np.zeros([nCases, self.nw])
        self.results['case_metrics']['wave_PSD1'] = np.zeros([nCases, self.nw])
        self.results['case_metrics']['wave_PSD2'] = np.zeros([nCases, self.nw])

        # store Damage equivalant Load from bending moments
        self.results['case_metrics']['DEL_angles'] = np.zeros([nCases, self.numAngles])

        # calculate the system's constant properties
        for fowt in self.fowtList:
            fowt.calcStatics()
            # Calculate BEM for all wave headings included in cases
            self.caseHeadings, headingStep, numberOfHeadings = getUniqueCaseHeadings(self.design['cases']['keys'],
                                                                                     self.design['cases']['data'])
            minHeading = min(self.caseHeadings)
            if len(self.caseHeadings) == 2:
                fowt.calcBEM(nHeadings=nCases, minHeading=minHeading, headingStep=headingStep)
            elif len(self.caseHeadings) > 2:
                fowt.calcBEM(nHeadings=numberOfHeadings, minHeading=minHeading, headingStep=headingStep)
                # JvS: Consider adding interpolation later, to reduce number of evaluations
            else:
                fowt.calcBEM(nHeadings=1, minHeading=minHeading)

        # loop through each case

        def evaluateCases(iCase):
            # for iCase in range(nCases): # This is old code from running after each other
            # TODO: JvS: Indented print statements below because parallel computating picks up next task in line and printing cost time. Consider moving to if statement if number of cores allowed == 1.

            #     print(f"\n--------------------- Running Case {iCase+1} ----------------------")
            #     print(self.design['cases']['data'][iCase])
            # form dictionary of case parameters
            case = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][iCase]))

            # get initial FOWT values assuming no offset
            for fowt in self.fowtList:
                fowt.Xi0 = np.zeros(6)  # zero platform offsets
                fowt.calcTurbineConstants(case, ptfm_pitch=0.0)
                fowt.calcHydroConstants(case)

            # calculate platform offsets and mooring system equilibrium state
            self.calcMooringAndOffsets()

            # update values based on offsets if applicable

            for fowt in self.fowtList:
                # print('RMS value pitch and roll', (fowt.Xi0[4] ** 2 + fowt.Xi0[3] ** 2) ** 0.5)
                fowt.calcTurbineConstants(case, ptfm_pitch=(fowt.Xi0[4] ** 2 + fowt.Xi0[3] ** 2) ** 0.5)
                # fowt.calcHydroConstants(case)  (hydrodynamics don't account for offset, so far)

            # (could solve mooring and offsets a second time, but likely overkill)

            # ------------------------------------------------------------------------

            if nCases > 100:
                # solve system dynamics
                self.solveDynamics(case)

                gc.collect()
                return self.Xi[0:6, :], fowt.Xi0
            else:
                # solve system dynamics
                _, M_tot, B_tot, C_tot = self.solveDynamics(case)
                return self.fowtList, self.Xi[0:6,:], fowt.Xi0, self.J_moor, self.T_moor, M_tot, B_tot, C_tot, fowt.A_aero, fowt.B_aero

        print('Starting calculation of the load cases, if there are more then 500 cases, the complete fowt-class is not stored to save memory.')
        print('Therefore, some plot functions will return empty.')
        res = Parallel(n_jobs=numberOfCores, timeout=99999)(delayed(evaluateCases)(iCase) for iCase in tqdm(range(nCases)))
        print('Calculation of cases finish, start storing the results and post processing now.')

        if nCases > 100:
            print('More than 50 cases')
            Xi_store = [item[0] for item in res]
            Xi0_store = [item[1] for item in res]

        else:
            fowtList = [item[0] for item in res]
            Xi_store = [item[1] for item in res]
            Xi0_store = [item[2] for item in res]
            Mooring_C = [item[3] for item in res]
            Mooring_T = [item[4] for item in res]
            self.M_tot_store = [item[5] for item in res]
            self.B_tot_store = [item[6] for item in res]
            self.C_tot_store = [item[7] for item in res]
            self.fowt_A_aero_stored = [item[8] for item in res]
            self.fowt_B_aero_stored = [item[9] for item in res]

        # results_store = [item[4] for item in res]

        for iCase in range(nCases):
            self.Xi = Xi_store[iCase]

            if nCases <= 100:
                self.T_moor = Mooring_T[iCase]
                self.C_moor = Mooring_C[iCase]
                self.fowtList = fowtList[iCase]

            # form dictionary of case parameters
            case = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][iCase]))
            if nCases > 100:
                for fowt in self.fowtList:
                    fowt.Xi0 = Xi0_store[iCase]
                    fowt.calcTurbineConstants(case, ptfm_pitch=(fowt.Xi0[4] ** 2 + fowt.Xi0[3] ** 2) ** 0.5)
                    # fowt.calcHydroConstants(case)  (hydrodynamics don't account for offset, so far)

            # ------------------------------------------------------------------------
            # process outputs that are specific to the floating unit
            self.fowtList[0].saveTurbineOutputs(self.results['case_metrics'], case, iCase, fowt.Xi0, self.Xi[0:6, :])
            # self.fowtList[0].saveTurbineOutputs(self.results['case_metrics'], case, iCase, fowt.Xi0, self.Xi[0:6,:])

            # process mooring tension outputs
            nLine = int(len(self.T_moor) / 2)
            T_moor_amps = np.zeros([2 * nLine, self.nw], dtype=complex)
            for iw in range(self.nw):
                T_moor_amps[:, iw] = np.matmul(self.J_moor, self.Xi[:, iw])  # FFT of mooring tensions

            self.results['case_metrics']['Tmoor_avg'][iCase, :] = self.T_moor
            for iT in range(2 * nLine):
                TRMS = getRMS(T_moor_amps[iT, :], self.w[0])  # estimated mooring line RMS tension [N]
                self.results['case_metrics']['Tmoor_std'][iCase, iT] = TRMS
                self.results['case_metrics']['Tmoor_max'][iCase, iT] = self.T_moor[iT] + 3 * TRMS
                self.results['case_metrics']['Tmoor_PSD'][iCase, iT, :] = getPSD(
                    T_moor_amps[iT, :])  # PSD in N^2/(rad/s)
                # self.results['case_metrics']['Tmoor_DEL'][iCase,iT] =

            if display:

                metrics = self.results['case_metrics']

                # print statistics table
                print(f"-------------------- Case {iCase + 1} Statistics --------------------")
                print("Response channel     Average     RMS         Maximum")
                print(f"surge (m)          {metrics['surge_avg'][iCase] :10.2e}  {metrics['surge_std'][iCase] :10.2e}  {metrics['surge_max'][iCase] :10.2e}")
                print(f"sway (m)           {metrics['sway_avg'][iCase] :10.2e}  {metrics['sway_std'][iCase] :10.2e}  {metrics['sway_max'][iCase] :10.2e}")
                print(f"heave (m)          {metrics['heave_avg'][iCase] :10.2e}  {metrics['heave_std'][iCase] :10.2e}  {metrics['heave_max'][iCase] :10.2e}")
                print(f"roll (deg)         {metrics['roll_avg'][iCase] :10.2e}  {metrics['roll_std'][iCase] :10.2e}  {metrics['roll_max'][iCase] :10.2e}")
                print(f"pitch (deg)        {metrics['pitch_avg'][iCase] :10.2e}  {metrics['pitch_std'][iCase] :10.2e}  {metrics['pitch_max'][iCase] :10.2e}")
                print(f"yaw (deg)          {metrics['yaw_avg'][iCase] :10.2e}  {metrics['yaw_std'][iCase] :10.2e}  {metrics['yaw_max'][iCase] :10.2e}")
                print(f"nacelle acc. (m/s) {metrics['AxRNA_avg'][iCase] :10.2e}  {metrics['AxRNA_std'][iCase] :10.2e}  {metrics['AxRNA_max'][iCase] :10.2e}")
                print(f"tower bending (Nm) {metrics['Mbase_avg'][iCase] :10.2e}  {metrics['Mbase_std'][iCase] :10.2e}  {metrics['Mbase_max'][iCase] :10.2e}")
                print(f"tower bending SS(Nm) {metrics['MbaseSS_avg'][iCase] :10.2e}  {metrics['MbaseSS_std'][iCase] :10.2e}  {metrics['MbaseSS_max'][iCase] :10.2e}")
                print(f"rotor speed (RPM)  {metrics['omega_avg'][iCase] :10.2e}  {metrics['omega_std'][iCase] :10.2e}  {metrics['omega_max'][iCase] :10.2e}")
                print(f"blade pitch (deg)  {metrics['bPitch_avg'][iCase] :10.2e}  {metrics['bPitch_std'][iCase] :10.2e} ")
                print(f"rotor power        {metrics['power_avg'][iCase] :10.2e} ")
                print(f"rotor thrust       {metrics['thrust_avg'][iCase] :10.2e} ")

                for i in range(nLine):
                    j = i + nLine
                    # print(f"line {i} tension A  {metrics['Tmoor_avg'][iCase,i]:10.2e}  {metrics['Tmoor_std'][iCase,i]:10.2e}  {metrics['Tmoor_max'][iCase,i]:10.2e}")
                    print(
                        f"line {i} tension (N) {metrics['Tmoor_avg'][iCase, j]:10.2e}  {metrics['Tmoor_std'][iCase, j]:10.2e}  {metrics['Tmoor_max'][iCase, j]:10.2e}")
                print(f"-----------------------------------------------------------")

        # print('Jobs finished')

    """
    def calcSystemConstantProps(self):
        '''This gets the various static/constant calculations of each FOWT done. (Those that don't depend on load case.)'''

        for fowt in self.fowtList:
            fowt.calcBEM()
            fowt.calcStatics()
            #fowt.calcDynamicConstants()
        
        # First get mooring system characteristics about undisplaced platform position (useful for baseline and verification)
        try:
            self.C_moor0 = self.ms.getCoupledStiffness(lines_only=True)                             # this method accounts for eqiuilibrium of free objects in the system
            self.F_moor0 = self.ms.getForces(DOFtype="coupled", lines_only=True)
        except Exception as e:
            raise RuntimeError('An error occured when getting linearized mooring properties in undisplaced state: '+e.message)
        
        self.results['properties'] = {}   # signal this data is available by adding a section to the results dictionary
    """

    def calcMooringAndOffsets(self):
        '''Calculates mean offsets and linearized mooring properties for the current load case.
        setEnv and calcSystemProps must be called first.  This will ultimately become a method for solving mean operating point.
        '''

        # apply any mean aerodynamic and hydrodynamic loads
        F_PRP = self.fowtList[0].F_aero0  # + self.fowtList[0].F_hydro0 <<< hydro load would be nice here eventually
        self.ms.bodyList[0].f6Ext = np.array(F_PRP)

        # Now find static equilibrium offsets of platform and get mooring properties about that point
        # (This assumes some loads have been applied)
        # self.ms.display=2

        try:
            self.ms.solveEquilibrium3(DOFtype="both",
                                      tol=0.01)  # , rmsTol=1.0E-5)     # get the system to its equilibrium
        except Exception as e:  # mp.MoorPyError
            print('An error occured when solving system equilibrium: ' + e.message)
            # raise RuntimeError('An error occured when solving unloaded equilibrium: '+error.message)

        # ::: a loop could be added here for an array :::
        fowt = self.fowtList[0]

        # print("Equilibrium'3' platform positions/rotations:")
        # printVec(self.ms.bodyList[0].r6)

        r6eq = self.ms.bodyList[0].r6
        fowt.Xi0 = np.array(r6eq)  # save current mean offsets for the FOWT
        # self.ms.plot()

        if self.printStatements:
            print(f"Found mean offets with with surge = {r6eq[0]:.2f} m and pitch = {r6eq[4] * 180 / np.pi:.2f} deg.")

        try:
            C_moor, J_moor = self.ms.getCoupledStiffness(lines_only=True,
                                                         tensions=True)  # get stiffness matrix and tension jacobian matrix
            F_moor = self.ms.getForces(DOFtype="coupled",
                                       lines_only=True)  # get net forces and moments from mooring lines on Body
            T_moor = self.ms.getTensions()
        except Exception as e:
            raise RuntimeError(
                'An error occured when getting linearized mooring properties in offset state: ' + e.message)

        # add any additional yaw stiffness that isn't included in the MoorPy model (e.g. if a bridle isn't modeled)
        C_moor[5, 5] += fowt.yawstiff

        self.C_moor = C_moor
        self.J_moor = J_moor  # jacobian of mooring line tensions w.r.t. coupled DOFs
        self.F_moor = F_moor
        self.T_moor = T_moor

        # store results
        self.results['means'] = {}  # signal this data is available by adding a section to the results dictionary
        self.results['means']['aero force'] = self.fowtList[0].F_aero0
        self.results['means']['platform offset'] = r6eq
        self.results['means']['mooring force'] = F_moor
        self.results['means']['fairlead tensions'] = np.array(
            [np.linalg.norm(self.ms.pointList[id - 1].getForces()) for id in self.ms.bodyList[0].attachedP])

    def solveEigen(self):
        '''finds natural frequencies of system'''

        # total system coefficient arrays
        M_tot = np.zeros([self.nDOF, self.nDOF])  # total mass and added mass matrix [kg, kg-m, kg-m^2]
        C_tot = np.zeros([self.nDOF, self.nDOF])  # total stiffness matrix [N/m, N, N-m]

        # add in mooring stiffness from MoorPy system
        C_tot += np.array(self.C_moor0)

        # ::: a loop could be added here for an array :::
        fowt = self.fowtList[0]

        # add any additional yaw stiffness that isn't included in the MoorPy model (e.g. if a bridle isn't modeled)
        C_tot[5, 5] += fowt.yawstiff  # will need to be put in calcSystemProps() once there is more than 1 fowt in a model

        # add fowt's terms to system matrices (BEM arrays are not yet included here)
        M_tot += fowt.M_struc + fowt.A_hydro_morison  # mass
        C_tot += fowt.C_struc + fowt.C_hydro  # stiffness

        # check viability of matrices
        message = ''
        for i in range(self.nDOF):
            if M_tot[i, i] < 1.0:
                message += f'Diagonal entry {i} of system mass matrix is less than 1 ({M_tot[i, i]}). '
            if C_tot[i, i] < 1.0:
                message += f'Diagonal entry {i} of system stiffness matrix is less than 1 ({C_tot[i, i]}). '

        if len(message) > 0:
            raise RuntimeError(
                'System matrices computed by RAFT have one or more small or negative diagonals: ' + message)

        # calculate natural frequencies (using eigen analysis to get proper values for pitch and roll - otherwise would need to base about CG if using diagonal entries only)
        eigenvals, eigenvectors = np.linalg.eig(np.matmul(np.linalg.inv(M_tot),
                                                          C_tot))  # <<< need to sort this out so it gives desired modes, some are currently a bit messy

        if any(eigenvals <= 0.0):
            raise RuntimeError("Error: zero or negative system eigenvalues detected.")

        # sort to normal DOF order based on which DOF is largest in each eigenvector
        ind_list = []
        for i in range(5, -1, -1):
            vec = np.abs(eigenvectors[i,
                         :])  # look at each row (DOF) at a time (use reverse order to pick out rotational DOFs first)

            for j in range(6):  # now do another loop in case the index was claimed previously

                ind = np.argmax(vec)  # find the index of the vector with the largest value of the current DOF

                if ind in ind_list:  # if a previous vector claimed this DOF, set it to zero in this vector so that we look at the other vectors
                    vec[ind] = 0.0
                else:
                    ind_list.append(ind)  # if it hasn't been claimed before, assign this vector to the DOF
                    break

        ind_list.reverse()  # reverse the index list since we made it in reverse order

        fns = np.sqrt(
            eigenvals[ind_list]) / 2.0 / np.pi  # apply sorting to eigenvalues and convert to natural frequency in Hz
        modes = eigenvectors[:, ind_list]  # apply sorting to eigenvectors

        print("")
        print("--------- Natural frequencies and mode shapes -------------")
        print("Mode        1         2         3         4         5         6")
        print("Fn (Hz)" + "".join([f"{fn:10.4f}" for fn in fns]))
        print("")
        for i in range(6):
            print(f"DOF {i + 1}  " + "".join([f"{modes[i, j]:10.4f}" for j in range(6)]))
        print("-----------------------------------------------------------")

        '''
        print("natural frequencies from eigen values")
        printVec(fns)
        print(1/fns)
        print("mode shapes from eigen values")
        printMat(modes)
        
        # alternative attempt to calculate natural frequencies based on diagonal entries (and taking pitch and roll about CG)
        if C_tot[0,0] == 0.0:
            zMoorx = 0.0
        else:
            zMoorx = C_tot[0,4]/C_tot[0,0]  # effective z elevation of mooring system reaction forces in x and y directions

        if C_tot[1,1] == 0.0:
            zMoory = 0.0
        else:
            zMoory = C_tot[1,3]/C_tot[1,1]

        zCG  = fowt.rCG_TOT[2]                    # center of mass in z
        zCMx = M_tot[0,4]/M_tot[0,0]              # effective z elevation of center of mass and added mass in x and y directions
        zCMy = M_tot[1,3]/M_tot[1,1]

        print("natural frequencies with added mass")
        fn = np.zeros(6)
        fn[0] = np.sqrt( C_tot[0,0] / M_tot[0,0] )/ 2.0/np.pi
        fn[1] = np.sqrt( C_tot[1,1] / M_tot[1,1] )/ 2.0/np.pi
        fn[2] = np.sqrt( C_tot[2,2] / M_tot[2,2] )/ 2.0/np.pi
        fn[5] = np.sqrt( C_tot[5,5] / M_tot[5,5] )/ 2.0/np.pi
        fn[3] = np.sqrt( (C_tot[3,3] + C_tot[1,1]*((zCMy-zMoory)**2 - zMoory**2) ) / (M_tot[3,3] - M_tot[1,1]*zCMy**2 ))/ 2.0/np.pi     # this contains adjustments to reflect rotation about the CG rather than PRP
        fn[4] = np.sqrt( (C_tot[4,4] + C_tot[0,0]*((zCMx-zMoorx)**2 - zMoorx**2) ) / (M_tot[4,4] - M_tot[0,0]*zCMx**2 ))/ 2.0/np.pi     # this contains adjustments to reflect rotation about the CG rather than PRP
        # note that the above lines use off-diagonal term rather than parallel axis theorem since rotation will not be exactly at CG due to effect of added mass
        printVec(fn)
        print(1/fn)
        '''

        # store results
        self.results['eigen'] = {}  # signal this data is available by adding a section to the results dictionary
        self.results['eigen']['frequencies'] = fns
        self.results['eigen']['modes'] = modes

    def solveDynamics(self, case, tol=0.01, conv_plot=0, RAO_plot=0, F_BEM_plot=False):
        '''After all constant parts have been computed, call this to iterate through remaining terms
        until convergence on dynamic response. Note that steady/mean quantities are excluded here.

        nIter = 2  # maximum number of iterations to allow
        '''

        nIter = int(self.nIter) + 1  # maybe think of a better name for the first nIter
        XiStart = self.XiStart

        # total system complex response amplitudes (this gets updated each iteration)
        XiLast = np.zeros([self.nDOF, self.nw],
                          dtype=complex) + XiStart  # displacement and rotation complex amplitudes [m, rad]

        if conv_plot:
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=get_figsize(self.latex_width, subplots=(3,1)))
            c = np.arange(nIter + 1)  # adding 1 again here so that there are no RuntimeErrors
            c = cm.jet((c - np.min(c)) / (
                    np.max(c) - np.min(c)))  # set up colormap to use to plot successive iteration results

        # ::: a loop could be added here for an array :::
        fowt = self.fowtList[0]
        i1 = 0  # range of DOFs for the current turbine
        i2 = 6

        # TEMPORARY <<<<
        # fowt.B_aero[0,4,:] = 0.0
        # fowt.B_aero[4,0,:] = 0.0
        if case['wind_heading'] != 0:
            fowt.M_struc = rotateMatrix6(fowt.M_struc,rotationMatrix(0,0,np.deg2rad(case['wind_heading'])))

            # fowt.M_struc[3,3] = fowt.M_struc[3,3]*np.cos(np.deg2rad(case['wind_heading']))\
            #                         +fowt.M_struc[4, 4] * np.sin(np.deg2rad(case['wind_heading']))
            # fowt.M_struc[4, 4] = fowt.M_struc[4, 4] * np.cos(np.deg2rad(case['wind_heading']))\
            #                         + fowt.M_struc[3, 3] * np.sin(np.deg2rad(case['wind_heading']))
        # sum up all linear (non-varying) matrices up front
        # print(fowt.M_struc)
        # with np.printoptions(precision=2, suppress=True):
        #     print(f'C_moor')
        #     print(bmatrix(self.C_moor))
        #
        #     print(f'C_struc')
        #     print(bmatrix(fowt.C_struc))
        #
        #     print(f'C_hydro')
        #     print(bmatrix(fowt.C_hydro))
        # print(f'C_struc {fowt.C_struc[:, :]}')
        # print(f'C_hydro {fowt.C_hydro[:, :]}')
        M_lin = fowt.A_aero + fowt.M_struc[:, :, None] + fowt.A_BEM + fowt.A_hydro_morison[:, :, None]  # mass
        B_lin = fowt.B_aero + fowt.B_struc[:, :, None] + fowt.B_BEM  # damping
        C_lin = fowt.C_struc + self.C_moor + fowt.C_hydro  # stiffness
        F_lin = fowt.F_aero + fowt.F_BEM + fowt.F_hydro_iner  # excitation

        if F_BEM_plot:
            freqMesh, headingMesh = np.meshgrid(fowt.w/TwoPi, fowt.headsStored[:38])
            figF, axF = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=get_figsize(self.latex_width, subplots=(3,2)))
            CB1 = axF[0, 0].contourf(freqMesh, headingMesh, np.abs(fowt.F_BEM_ALL[:38, 0, :]))  # surge
            CB2 = axF[1, 0].contourf(freqMesh, headingMesh, np.abs(fowt.F_BEM_ALL[:38, 1, :]))
            CB3 = axF[2, 0].contourf(freqMesh, headingMesh, np.abs(fowt.F_BEM_ALL[:38, 2, :]))
            CB4 = axF[0, 1].contourf(freqMesh, headingMesh, np.abs(fowt.F_BEM_ALL[:38, 3, :]))
            CB5 = axF[1, 1].contourf(freqMesh, headingMesh, np.abs(fowt.F_BEM_ALL[:38, 4, :]))
            CB6 = axF[2, 1].contourf(freqMesh, headingMesh, np.abs(fowt.F_BEM_ALL[:38, 5, :]))
            axF[0, 0].set_title('Surge Wave Excitation')
            axF[1, 0].set_title('Sway Wave Excitation')
            axF[2, 0].set_title('Heave Wave Excitation')
            axF[0, 1].set_title('Roll Wave Excitation')
            axF[1, 1].set_title('Pitch Wave Excitation')
            axF[2, 1].set_title('Yaw Wave Excitation')
            axF[2, 0].set_xlabel('Frequency [Hz]')
            axF[2, 1].set_xlabel('Frequency [Hz]')
            axF[0, 0].set_ylabel('Wave Heading [deg]')
            axF[1, 0].set_ylabel('Wave Heading [deg]')
            axF[2, 0].set_ylabel('Wave Heading [deg]')

            # figF.colorbar(CB1, ax=axF[0, 0])
            # figF.colorbar(CB2, ax=axF[1, 0])
            # figF.colorbar(CB3, ax=axF[2, 0])
            # figF.colorbar(CB4, ax=axF[0, 1])
            # figF.colorbar(CB5, ax=axF[1, 1])
            # figF.colorbar(CB6, ax=axF[2, 1])
            cbar1 = figF.colorbar(CB1, ax=axF[0, 0])
            cbar2 = figF.colorbar(CB2, ax=axF[1, 0])
            cbar3 = figF.colorbar(CB3, ax=axF[2, 0])
            cbar4 = figF.colorbar(CB4, ax=axF[0, 1])
            cbar5 = figF.colorbar(CB5, ax=axF[1, 1])
            cbar6 = figF.colorbar(CB6, ax=axF[2, 1])

            cbar1.ax.set_ylabel('Force [N]')
            cbar2.ax.set_ylabel('Force [N]')
            cbar3.ax.set_ylabel('Force [N]')
            cbar4.ax.set_ylabel('Moment [Nm]')
            cbar5.ax.set_ylabel('Moment [Nm]')
            cbar6.ax.set_ylabel('Moment [Nm]')



            if case['add_waveheading'] == True:
                hs = case['wave_height2']
                tp = case['wave_period2']
            else:
                hs = case['wave_height']
                tp = case['wave_period']
            figF.suptitle(f'Real part of wave excitation for various wave headings ($H_s$ = {hs} [m], $T_p$ = {tp} [s])')
            figF.savefig(f'F_BEM_hs_{hs}_m_tp_{tp}_sec.pdf')
            # figF.savefig(f'F_BEM_hs_{hs}_m_tp_{tp}_sec.pdf')
            # for col in range(2):
            #     for row in range(3):
            #         figF.colorbar(axF, ax = axF[row,col])
            # freqMesh, headingMesh = np.meshgrid(self.w,self.headsStored)
            # figAng, axAng = plt.subplots(3,2)
            # axAng[0,0].contour(freqMesh,headingMesh,self.X_BEM[:,0,:])

            # start fixed point iteration loop for dynamics
        for iiter in range(nIter):

            # initialize/zero total system coefficient arrays
            M_tot = np.zeros([self.nDOF, self.nDOF, self.nw])  # total mass and added mass matrix [kg, kg-m, kg-m^2]
            B_tot = np.zeros([self.nDOF, self.nDOF, self.nw])  # total damping matrix [N-s/m, N-s, N-s-m]
            C_tot = np.zeros([self.nDOF, self.nDOF, self.nw])  # total stiffness matrix [N/m, N, N-m]
            F_tot = np.zeros([self.nDOF, self.nw],
                             dtype=complex)  # total excitation force/moment complex amplitudes vector [N, N-m]

            Z = np.zeros([self.nDOF, self.nDOF, self.nw], dtype=complex)  # total system impedance matrix

            # a loop could be added here for an array
            fowt = self.fowtList[0]

            # get linearized terms for the current turbine given latest amplitudes
            B_linearized, F_linearized = fowt.calcLinearizedTerms(XiLast)

            # calculate the response based on the latest linearized terms
            Xi = np.zeros([self.nDOF, self.nw], dtype=complex)  # displacement and rotation complex amplitudes [m, rad]

            # add fowt's terms to system matrices (BEM arrays are not yet included here)
            M_tot[:,:,:] = M_lin
            B_tot[:,:,:] = B_lin           + B_linearized[:,:,None]
            C_tot[:,:,:] = C_lin[:,:,None]
            F_tot[:  ,:] = F_lin           + F_linearized

            for ii in range(self.nw):
                # form impedance matrix
                Z[:, :, ii] = -self.w[ii] ** 2 * M_tot[:, :, ii] + 1j * self.w[ii] * B_tot[:, :, ii] + C_tot[:, :, ii]

                # solve response (complex amplitude)
                # Xi[:, ii] = np.matmul(np.linalg.inv(Z[:, :, ii]), F_tot[:, ii])
                Xi[:, ii] = np.linalg.solve(Z[:, :, ii], F_tot[:, ii])
            if conv_plot:
                # Convergence Plotting
                # plots of surge response at each iteration for observing convergence
                ax[0].plot(self.w, np.abs(Xi[0, :]), color=c[iiter], label=f"iteration {iiter}")
                ax[1].plot(self.w, np.real(Xi[0, :]), color=c[iiter], label=f"iteration {iiter}")
                ax[2].plot(self.w, np.imag(Xi[0, :]), color=c[iiter], label=f"iteration {iiter}")

            # check for convergence
            tolCheck = np.abs(Xi - XiLast) / ((np.abs(Xi) + tol))
            if (tolCheck < tol).all():
                if self.printStatements:
                    print(f" Iteration {iiter}, converged (largest change is {np.max(tolCheck):.5f} < {tol})")
                break
            else:
                XiLast = 0.2 * XiLast + 0.8 * Xi  # use a mix of the old and new response amplitudes to use for the next iteration
                # (uses hard-coded successive under relaxation for now)
                if self.printStatements:
                    print(f" Iteration {iiter}, unconverged (largest change is {np.max(tolCheck):.5f} >= {tol})")

            if iiter == nIter - 1:
                print("WARNING - solveDynamics iteration did not converge to the tolerance.")

        if conv_plot:
            # labels for convergence plots
            ax[1].legend()
            ax[0].set_ylabel("response magnitude")
            ax[1].set_ylabel("response, real")
            ax[2].set_ylabel("response, imag")
            ax[2].set_xlabel("frequency (rad/s)")
            fig.suptitle("Response convergence")

        # ------------------------------ preliminary plotting of response ---------------------------------

        if RAO_plot:
            # RAO plotting
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=get_figsize(self.latex_width, subplots=(3,1)))

            fowt = self.fowtList[0]

            ax[0].plot(self.w, np.abs(Xi[0, :]), 'b', label="surge")
            ax[0].plot(self.w, np.abs(Xi[1, :]), 'g', label="sway")
            ax[0].plot(self.w, np.abs(Xi[2, :]), 'r', label="heave")
            ax[1].plot(self.w, np.abs(Xi[3, :]) * 180 / np.pi, 'b', label="roll")
            ax[1].plot(self.w, np.abs(Xi[4, :]) * 180 / np.pi, 'g', label="pitch")
            ax[1].plot(self.w, np.abs(Xi[5, :]) * 180 / np.pi, 'r', label="yaw")
            ax[2].plot(self.w, fowt.storeZeta[0, :], 'k', label="wave amplitude 1 (m)")
            if np.all(fowt.storeZeta[1, :] == 0) == False:
                ax[2].plot(self.w, fowt.storeZeta[1, :], 'k', label="wave amplitude 2 (m)")

            ax[0].legend()
            ax[1].legend()
            ax[2].legend()

            # ax[0].set_ylim([0, 1e6])
            # ax[1].set_ylim([0, 1e9])

            ax[0].set_ylabel("response magnitude (m)")
            ax[1].set_ylabel("response magnitude (deg)")
            ax[2].set_ylabel("wave amplitude (m)")
            ax[2].set_xlabel("frequency (rad/s)")



        if F_BEM_plot:
            i = 0
            fig, ax = plt.subplots(3,2, sharex=True, figsize=get_figsize(self.latex_width, subplots=(3,2)))
            for m in [0, 1]:
                for n in [0,1,2]:
                    ax[n,m].plot(self.w /TwoPi, fowt.F_aero[i, :], label='Aerodynamic excitation')
                    ax[n,m].plot(self.w /TwoPi, fowt.F_BEM[i, :], label='Potential flow excitation')
                    ax[n,m].plot(self.w /TwoPi, fowt.F_hydro_iner[i, :], label='Hydro inertia excitation')
                    ax[n,m].plot(self.w /TwoPi, fowt.F_hydro_drag[i, :], label='Hydrodynamic drag excitation')
                    i += 1

            ax[0, 0].set_title('Surge Direction')
            ax[1, 0].set_title('Sway Direction')
            ax[2, 0].set_title('Heave Direction')
            ax[0, 1].set_title('Roll Direction')
            ax[1, 1].set_title('Pitch Direction')
            ax[2, 1].set_title('Yaw Direction')

            ax[0, 0].set_ylabel('Force [N]')
            ax[1, 0].set_ylabel('Force [N]')
            ax[2, 0].set_ylabel('Force [N]')
            ax[0, 1].set_ylabel('Moment [Nm]')
            ax[1, 1].set_ylabel('Moment [Nm]')
            ax[2, 1].set_ylabel('Moment [Nm]')

            ax[2, 0].set_xlabel('Frequency [Hz]')
            ax[2, 1].set_xlabel('Frequency [Hz]')

            ax[2,1].legend()
            fig.savefig('F_all.pdf')

        self.Xi = Xi

        self.results['response'] = {}  # signal this data is available by adding a section to the results dictionary

        # with np.printoptions(precision=2, suppress=True):
        #     print(f'B_drag')
        #     print(bmatrix(B_linearized))

        return Xi, M_tot, B_tot, C_tot  # currently returning the response rather than saving in the model object

    def calcOutputs(self):
        '''This is where various output quantities of interest are calculated based on the already-solved system response.'''

        fowt = self.fowtList[0]  # just using a single turbine for now

        # ----- system properties outputs -----------------------------
        # all values about platform reference point (z=0) unless otherwise noted

        if 'properties' in self.results:
            self.results['properties']['tower mass'] = fowt.mtower
            self.results['properties']['tower CG'] = fowt.rCG_tow
            self.results['properties']['substructure mass'] = fowt.msubstruc
            self.results['properties']['substructure CG'] = fowt.rCG_sub
            self.results['properties']['shell mass'] = fowt.mshell
            self.results['properties']['ballast mass'] = fowt.mballast
            self.results['properties']['ballast densities'] = fowt.pb
            self.results['properties']['total mass'] = fowt.M_struc[0, 0]
            self.results['properties']['total CG'] = fowt.rCG_TOT
            # self.results['properties']['roll inertia at subCG'] = fowt.I44
            # self.results['properties']['pitch inertia at subCG'] = fowt.I55
            # self.results['properties']['yaw inertia at subCG'] = fowt.I66
            self.results['properties']['roll inertia at subCG'] = fowt.M_struc_subCM[3, 3]
            self.results['properties']['pitch inertia at subCG'] = fowt.M_struc_subCM[4, 4]
            self.results['properties']['yaw inertia at subCG'] = fowt.M_struc_subCM[5, 5]

            self.results['properties']['Buoyancy (pgV)'] = fowt.rho_water * fowt.g * fowt.V
            self.results['properties']['Center of Buoyancy'] = fowt.rCB
            self.results['properties']['C stiffness matrix'] = fowt.C_hydro

            # unloaded equilibrium <<< 

            self.results['properties']['F_lines0'] = self.F_moor0
            self.results['properties']['C_lines0'] = self.C_moor0

            # 6DOF matrices for the support structure (everything but turbine) including mass, hydrostatics, and mooring reactions
            self.results['properties']['M support structure'] = fowt.M_struc_subCM  # mass matrix
            self.results['properties']['A support structure'] = fowt.A_hydro_morison + fowt.A_BEM[:, :,
                                                                                       -1]  # hydrodynamic added mass (currently using highest frequency of BEM added mass)
            self.results['properties'][
                'C support structure'] = fowt.C_struc_sub + fowt.C_hydro + self.C_moor0  # stiffness

        # ----- response outputs (always in standard units) ---------------------------------------

        if 'response' in self.results:
            RAOmag = abs(self.Xi / fowt.zeta)  # magnitudes of motion RAO

            self.results['response']['frequencies'] = self.w / 2 / np.pi  # Hz
            self.results['response']['wave elevation'] = fowt.zeta
            self.results['response']['Xi'] = self.Xi
            self.results['response']['surge RAO'] = RAOmag[0, :]
            self.results['response']['sway RAO'] = RAOmag[1, :]
            self.results['response']['heave RAO'] = RAOmag[2, :]
            self.results['response']['pitch RAO'] = RAOmag[3, :]
            self.results['response']['roll RAO'] = RAOmag[4, :]
            self.results['response']['yaw RAO'] = RAOmag[5, :]

            # save dynamic derived quantities
            # self.results['response']['mooring tensions'] = ...
            self.results['response']['nacelle acceleration'] = self.w ** 2 * (self.Xi[0] + self.Xi[4] * fowt.hHub)

        return self.results

    def plotResponses(self):
        '''Plots the power spectral densities of the available response channels for each case.'''

        fig, ax = plt.subplots(6, 1, sharex=True, figsize=get_figsize(self.latex_width, subplots=(6,1)))

        metrics = self.results['case_metrics']
        nCases = len(metrics['surge_avg'])

        for iCase in range(nCases):
            ax[0].plot(self.w / TwoPi, TwoPi * metrics['surge_PSD'][iCase, :])  # surge
            ax[1].plot(self.w / TwoPi, TwoPi * metrics['heave_PSD'][iCase, :])  # heave
            ax[2].plot(self.w / TwoPi, TwoPi * metrics['pitch_PSD'][iCase, :])  # pitch [deg]
            ax[3].plot(self.w / TwoPi, TwoPi * metrics['AxRNA_PSD'][iCase, :])  # nacelle acceleration
            ax[4].plot(self.w / TwoPi,
                       TwoPi * metrics['Mbase_PSD'][iCase, :])  # tower base bending moment (using FAST's kN-m)
            ax[5].plot(self.w / TwoPi, TwoPi * metrics['wave_PSD1'][iCase, :],
                       label=f'case {iCase + 1}')  # wave spectrum

            # need a variable number of subplots for the mooring lines
            # ax2[3].plot(model.w/2/np.pi, TwoPi*metrics['Tmoor_PSD'][0,3,:]  )  # fairlead tension

        ax[0].set_ylabel('surge \n' + r'(m$^2$/Hz)')
        ax[1].set_ylabel('heave \n' + r'(m$^2$/Hz)')
        ax[2].set_ylabel('pitch \n' + r'(deg$^2$/Hz)')
        ax[3].set_ylabel('nac. acc. \n' + r'((m/s$^2$)$^2$/Hz)')
        ax[4].set_ylabel('twr. bend \n' + r'((Nm)$^2$/Hz)')
        ax[5].set_ylabel('wave elev.\n' + r'(m$^2$/Hz)')

        # ax[0].set_ylim([0.0, 25])
        # ax[1].set_ylim([0.0, 15])
        # ax[2].set_ylim([0.0, 4])
        # ax[-1].set_xlim([0.03, 0.15])
        ax[-1].set_xlabel('frequency (Hz)')

        # if nCases > 1:
        ax[-1].legend()
        fig.suptitle('RAFT power spectral densities')

    def plotResponses_extended(self):
        '''Plots more power spectral densities of the available response channels for each case.'''

        fig, ax = plt.subplot_mosaic([
            ['surge', 'AxRNA'],
            ['sway', 'MBaseFA'],
            ['heave', 'MBaseSS'],
            ['pitch', 'Wave1'],
            ['roll', 'Wave2'],
            ['yaw', 'Wind']
        ], figsize=get_figsize(self.latex_width, subplots=(6,2)))  # subplots(11, 1, sharex=True)

        metrics = self.results['case_metrics']
        nCases = len(metrics['surge_avg'])

        for iCase in range(nCases):
            case = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][iCase]))

            ax['surge'].plot(self.w / TwoPi, TwoPi * metrics['surge_PSD'][iCase, :])  # surge
            ax['sway'].plot(self.w / TwoPi, TwoPi * metrics['sway_PSD'][iCase, :])  # surge
            ax['heave'].plot(self.w / TwoPi, TwoPi * metrics['heave_PSD'][iCase, :])  # heave
            ax['pitch'].plot(self.w / TwoPi, TwoPi * metrics['pitch_PSD'][iCase, :])  # pitch [deg]
            ax['roll'].plot(self.w / TwoPi, TwoPi * metrics['roll_PSD'][iCase, :])  # pitch [deg]
            ax['yaw'].plot(self.w / TwoPi, TwoPi * metrics['yaw_PSD'][iCase, :])  # pitch [deg]
            ax['AxRNA'].plot(self.w / TwoPi, TwoPi * metrics['AxRNA_PSD'][iCase, :])  # nacelle acceleration
            ax['MBaseFA'].plot(self.w / TwoPi,
                               TwoPi * metrics['Mbase_PSD'][iCase, :])  # tower base bending moment (using FAST's kN-m)
            ax['MBaseSS'].plot(self.w / TwoPi, TwoPi * metrics['MbaseSS_PSD'][iCase,
                                                       :])  # tower base bending moment (using FAST's kN-m)
            ax['Wave1'].plot(self.w / TwoPi, TwoPi * metrics['wave_PSD1'][iCase, :],
                             label=f'ws 1 case {iCase + 1}')  # wave spectrum
            if not np.all(metrics['wave_PSD2'][iCase, :] == 0):
                ax['Wave2'].plot(self.w / TwoPi, TwoPi * metrics['wave_PSD2'][iCase, :],
                                 label=f'ws 2 case {iCase + 1}')  # wave spectrum
            case_wh = case['wave_heading2']
            ax['Wind'].plot(self.w / TwoPi, TwoPi * metrics['wind_PSD'][iCase, :],
                            label=f'Misalignment swell wave = {case_wh} [deg]')  # wind spectrum
            # need a variable number of subplots for the mooring lines
            # ax2[3].plot(model.w/2/np.pi, TwoPi*metrics['Tmoor_PSD'][0,3,:]  )  # fairlead tension

        ax['surge'].set_ylabel('surge \n' + r'(m$^2$/Hz)')
        ax['sway'].set_ylabel('sway \n' + r'(m$^2$/Hz)')
        ax['heave'].set_ylabel('heave \n' + r'(m$^2$/Hz)')
        ax['pitch'].set_ylabel('pitch \n' + r'(deg$^2$/Hz)')
        ax['roll'].set_ylabel('roll \n' + r'(deg$^2$/Hz)')
        ax['yaw'].set_ylabel('yaw \n' + r'(deg$^2$/Hz)')
        ax['AxRNA'].set_ylabel('nac. acc. \n' + r'((m/s$^2$)$^2$/Hz)')
        ax['MBaseFA'].set_ylabel('twr. bend FA \n' + r'((Nm)$^2$/Hz)')
        ax['MBaseSS'].set_ylabel('twr. bend SS \n' + r'((Nm)$^2$/Hz)')
        ax['Wave1'].set_ylabel('wave elev.\n' + r'(m$^2$/Hz)')
        ax['Wave2'].set_ylabel('wave elev.\n' + r'(m$^2$/Hz)')
        ax['Wind'].set_ylabel('wind speed.\n' + r'((m/s)$^2$/Hz)')

        # ax[0].set_ylim([0.0, 25])
        # ax[1].set_ylim([0.0, 15])
        # ax[2].set_ylim([0.0, 4])
        # ax[-1].set_xlim([0.03, 0.15])
        ax['yaw'].set_xlabel('frequency [Hz]')
        ax['Wind'].set_xlabel('frequency [Hz]')

        # if nCases > 1:
        ax['Wind'].legend()
        ax['Wave1'].legend()
        ax['Wave2'].legend()
        # ax[''].legend()

        fig.suptitle('RAFT power spectral densities')
        fig.tight_layout()

    def plotTowerBaseResponse(self, plot='polar', plot_eq_stress_angles=False, include_surface=False, weighted_sum_cases = False):
        metrics = self.results['case_metrics']
        nCases = len(metrics['surge_avg'])

        for iCase in range(nCases):

            sigmaX, ANGLESMesh, FREQMesh = getSigmaXPSD(TBFA=metrics['Mbase_sig'][iCase, :],
                                                        TBSS=metrics['MbaseSS_sig'][iCase, :],
                                                        frequencies=self.w,
                                                        angles=self.anglesArray,
                                                        d= getFromDict(self.design['turbine']['tower'],'d', shape=-1,default=[10, 8])[0],
                                                        thickness= getFromDict(self.design['turbine']['tower'],'t', shape=-1,default=[0.1, 0.08])[0]
                                                        )

            DK_ps = np.zeros([self.numAngles, 40])
            if include_surface:
                plt.figure(figsize=get_figsize(self.latex_width))
                ax = plt.axes(projection='3d')
                ax.plot_surface(np.rad2deg(ANGLESMesh), FREQMesh / TwoPi, TwoPi * sigmaX,
                                cmap=plt.cm.jet)  # , rstride=1)
                ax.set_xlabel('angle around TB (deg)')
                ax.set_ylabel('frequency [Hz]')
                ax.set_zlabel('sigma_x (MPa^2/Hz)')
                ax.set_xbound(0, 360)
                ax.set_ybound(0, 0.4)

            for iAngles in range(self.numAngles):
                #     DK[iAngles] = Dirlik.DK(5.56, 1.62*10**22, sigmaX[iAngles,:], self.w, 0, 0 )
                #     print(DK[iAngles].Damage())
                moments = prob_moment.Probability_Moment(sigmaX[:, iAngles], self.w)
                sigma_max = 6 * moments.momentn(0) ** 0.5
                stress_range = np.linspace(0, sigma_max, 40)
                DK_temp = Dirlik.DK(6, 1.62 * 10 ** 22, sigmaX[:, iAngles], self.w, 30 * 364 * 24 * 3600, stress_range)
                DK_ps[iAngles, :] = DK_temp.counting_cycles()
                metrics['DEL_angles'][iCase, iAngles] = (np.dot(DK_temp.counting_cycles(), stress_range ** 6) / np.sum(
                    DK_temp.counting_cycles())) ** (1 / 6)

        if plot == 'polar':
            plt.figure(figsize=get_figsize(self.latex_width))
            ax = plt.subplot(111, polar=True)
            ax.grid(True)
            ax.set_theta_direction(-1)
            ax.set_theta_offset(np.pi / 2.0)
            for iCase in range(nCases):
                case = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][iCase]))
                wind_heading = float(case['wind_heading'])
                wave_1 = float(case['wave_heading'])
                if case['add_waveheading']:
                    wave_2 = float(case['wave_heading2'])
                    wave2string = f', wave 2 = {wave_2} deg'
                else:
                    wave2string = ''
                ax.plot(self.anglesArray, metrics['DEL_angles'][iCase, :],
                        label=f'Case {iCase + 1}, wind heading = {wind_heading} deg, wave 1 = {wave_1} deg{wave2string}')
            ax.set_title('Fatigue Damage Equivalant Load Around TB [MPa]')
            ax.set_xlabel('Location around TB circumference (deg)')
            if nCases <= 8:
                ax.legend()
        elif plot == 'angles':
            plt.figure(figsize=get_figsize(self.latex_width))
            plt.ylabel('Sigma_x (MPa)')
            plt.xlim([0, 360])
            for iCase in range(nCases):
                plt.plot(np.rad2deg(self.anglesArray), metrics['DEL_angles'][iCase, :], label=f'Case {iCase + 1}, ')
            plt.legend()
            plt.xlabel('Location around TB circumference (deg)')
            plt.title('Fatigue Damage Equivalant Load Around TB [MPa]')
            plt.grid()

        if plot_eq_stress_angles:
            variableXaxis = []
            fig, ax = plt.subplots(figsize=get_figsize(self.latex_width))
            for iCase in range(nCases):
                cases = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][iCase]))
                variableXaxis, string_x_axis, _ = retrieveAxisParAnalysis(iCase, cases, self.changeType, variableXaxis,
                                                                       self.design['parametricAnalysis'])
            ax.plot(variableXaxis, np.amax(metrics['DEL_angles'][:, :], 1), label='Max eq stress')
            ax.plot(variableXaxis, np.amin(metrics['DEL_angles'][:, :], 1), label='Min eq stress')
            ax.set_xlabel(string_x_axis)
            ax.set_ylabel('Stress (FDEL) MPa')
            ax.set_ylim(bottom=0)
            ax.set_title(f'Equivalent stress for changing {string_x_axis}')
            ax.legend()
            ax.grid()

        case = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][0]))
        if weighted_sum_cases and case['FLS_weight_factor'] is not None:
            print('Calculating weighted equivalent stress for all cases now...')
            total_weight_factor = 0
            for iCase in range(nCases):
                case = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][iCase]))
                total_weight_factor += float(case['FLS_weight_factor'])
            print(total_weight_factor)
            weightedEquivalentStressAllLoadCases = np.zeros(self.numAngles)
            counter_term = np.zeros(self.numAngles)

            for iAngles in range(self.numAngles):
                for iCase in range(nCases):

                    case = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][iCase]))

                    equivalent_stress_angle = metrics['DEL_angles'][iCase,iAngles]
                    counter_term[iAngles] += equivalent_stress_angle**6*float(case['FLS_weight_factor'])

                weightedEquivalentStressAllLoadCases[iAngles] = ((counter_term[iAngles])/total_weight_factor)**(1/6)

            plt.figure(figsize=get_figsize(self.latex_width))
            ax = plt.subplot(111, polar=True)
            ax.grid(True)
            ax.set_theta_direction(-1)
            ax.set_theta_offset(np.pi / 2.0)
            ax.plot(self.anglesArray, weightedEquivalentStressAllLoadCases, label=f'All Cases')
            ax.set_title('Fatigue Damage Equivalant Stress Around TB [MPa]')
            ax.set_xlabel('Location around TB circumference (deg)')
        elif case['FLS_weight_factor'] is None:
            print('case[FLS_weight_factor] not defined, skipping this step')




        # TODO: Seperate calculation of FDEL and Plotting in seperate methods, they can then be added to  results print.

        # fig = plt.figure(figsize=(8, 8))
        # ax1 = fig.add_subplot()
        # stressMesh, angMesh = np.meshgrid(stress_range, anglesvec)
        # pos = ax1.imshow(DK_ps,
        #             cmap='rainbow',
        #             interpolation='nearest',
        #             extent = [stress_range[0],stress_range[-1],anglesvec[0],anglesvec[-1]],
        #             aspect = stress_range[-1]/anglesvec[-1])#bar3d(angMesh.flatten(),
        # fig.colorbar(pos)
        # stressMesh.flatten(),
        # np.zeros_like(DK_ps.flatten()),
        # angMesh[1,1]-angMesh[0,0],
        # stressMesh[1,1]-stressMesh[0,0],
        # DK_ps.flatten())

    def RMSmisalignresponse(self, singleDOF=True, twoDOF=False, RootRMS=True):
        metrics = self.results['case_metrics']
        nCases = len(metrics['surge_avg'])
        rotation_RMS = []
        yawRMS = []
        rollRMS = []
        pitchRMS = []
        rollpitch = []
        pitchyaw = []
        rollyaw = []
        variableXaxis = []
        # print(dict(zip(self.design['cases']['keys'], self.design['cases']['data'])))
        for iCase in range(nCases):
            cases = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][iCase]))
            variablXaxis, string_x_axis, _ = retrieveAxisParAnalysis(iCase, cases, self.changeType, variableXaxis,
                                                                  self.design['parametricAnalysis'])
            root_value_yaw_roll = (metrics['roll_std'][iCase] ** 2 + metrics['yaw_std'][iCase] ** 2) ** 0.5
            root_value_yaw_pitch = (metrics['pitch_std'][iCase] ** 2 + metrics['yaw_std'][iCase] ** 2) ** 0.5
            root_value_roll_pitch = (metrics['roll_std'][iCase] ** 2 + metrics['pitch_std'][iCase] ** 2) ** 0.5

            root_value_all = (metrics['roll_std'][iCase] ** 2 + metrics['pitch_std'][iCase] ** 2 + metrics['yaw_std'][
                iCase] ** 2) ** 0.5
            rollRMS.append(metrics['roll_std'][iCase])
            pitchRMS.append(metrics['pitch_std'][iCase])
            yawRMS.append(metrics['yaw_std'][iCase])
            rollyaw.append(root_value_yaw_roll)
            pitchyaw.append(root_value_yaw_pitch)
            rollpitch.append(root_value_roll_pitch)
            rotation_RMS.append(root_value_all)

        fig, ax = plt.subplots(figsize=get_figsize(self.latex_width))
        if singleDOF:
            ax.plot(variableXaxis, rollRMS, label='RMS values of roll-response')
            ax.plot(variableXaxis, pitchRMS, label='RMS values of pitch-response')
            ax.plot(variableXaxis, yawRMS, label='RMS values of yaw-response')
        if twoDOF:
            ax.plot(variableXaxis, rollyaw, label='Root value of roll and yaw RMS')
            ax.plot(variableXaxis, pitchyaw, label='Root value of pitch and yaw RMS')
            ax.plot(variableXaxis, rollpitch, label='Root value of roll and pitch RMS')

        if RootRMS:
            ax.plot(variableXaxis, rotation_RMS, label='Root value of pitch, roll and yaw RMS')
        ax.set_ylim(bottom=0)
        ax.set_xlabel(string_x_axis)
        ax.set_ylabel('Root of RMS values [-]')
        ax.set_title('RMS values for rotational DOFs')
        ax.legend()
        ax.grid()

    # self.design['cases']['data'][iCase]

    def plotPowerThrust(self):
        variableXaxis = []
        metrics = self.results['case_metrics']
        nCases = len(metrics['surge_avg'])
        for iCase in range(nCases):
            cases = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][iCase]))
            variableXaxis, string_x_axis, _ = retrieveAxisParAnalysis(iCase, cases, self.changeType, variableXaxis,
                                                                   self.design['parametricAnalysis'])
        fig, ax1 = plt.subplots(figsize=get_figsize(self.latex_width))
        ax2 = ax1.twinx()
        ax1.plot(variableXaxis, metrics['power_avg'][:] / (10 ** 6), '-',label= 'Power')
        ax2.plot(variableXaxis, metrics['thrust_avg'][:] / (10 ** 6), '--', label = 'Thrust')
        ax1.set_xlabel(string_x_axis)
        ax1.set_ylabel('Generated Power [MW]', color='g')
        ax2.set_ylabel('Thrust on rotor [MW]', color='b')
        ax1.set_title('Turbine characteristics')
        ax1.legend()
        ax1.grid()

    def plotCouplingTerms(self, diagonal = True):
        diag_title = ' diagonal and'
        metrics = self.results['case_metrics']
        nCases = len(metrics['surge_avg'])

        variableXaxis = []
        DOF1 = [4, 6]
        DOF2 = [1, 6]

        for iCase in range(nCases):
            cases = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][iCase]))
            _, _, titleString = retrieveAxisParAnalysis(iCase, cases, self.changeType, variableXaxis,
                                                                   self.design['parametricAnalysis'])
            fig, ax = plt.subplots(3, 3, sharex=True, figsize=get_figsize(self.latex_width, subplots=(3,3)))
            for dof in range(DOF1[0] - 1, DOF1[1]):
                for dof2 in range(DOF2[0] - 1, DOF2[1]):
                    if dof == dof2 and diagonal is not True:
                        diag_title = ''
                    else:

                        ax[0, dof - 3].plot(self.w / TwoPi, self.M_tot_store[iCase][dof, dof2, :],
                                            label=f'$(M+A)_{{{dof + 1},{dof2 + 1}}}$')
                        ax[1, dof - 3].plot(self.w / TwoPi, self.B_tot_store[iCase][dof, dof2, :],
                                            label=f'$B_{{{dof + 1},{dof2 + 1}}}$')
                        ax[2, dof - 3].plot(self.w / TwoPi, self.C_tot_store[iCase][dof, dof2, :],
                                            label=f'$C_{{{dof + 1},{dof2 + 1}}}$')
                        ax[0, dof - 3].legend(loc = 1)
                        ax[1, dof - 3].legend(loc = 1)
                        ax[2, dof - 3].legend(loc = 1)

            ax[0, 0].set_title('Roll')
            ax[0, 1].set_title('Pitch')
            ax[0, 2].set_title('Yaw')
            ax[2, 0].set_xlabel('Frequency [Hz]')
            ax[2, 1].set_xlabel('Frequency [Hz]')
            ax[2, 2].set_xlabel('Frequency [Hz]')
            ax[0, 0].set_ylabel(f'$(M+A)$ $[kg m^2]$')
            ax[1, 0].set_ylabel(f'$B$ $[kg m^2/s]$')
            ax[2, 0].set_ylabel(f'$C$ $[kg m^2/s^2]$')
            fig.suptitle(f'Roll, Pitch, Yaw{diag_title} coupling terms ({titleString}).')

    def plotBEMTerms(self, diagonal=True):
        diag_title = ' diagonal and'

        fowt = self.fowtList[0]
        metrics = self.results['case_metrics']
        nCases = len(metrics['surge_avg'])
        DOF1 = [4, 6]
        DOF2 = [1, 6]
        fig, ax = plt.subplots(2, 3, sharex=True, figsize=get_figsize(self.latex_width, subplots=(2.2,3)))
        for dof in range(DOF1[0] - 1, DOF1[1]):
            for dof2 in range(DOF2[0] - 1, DOF2[1]):
                if dof==dof2 and diagonal is not True:
                    diag_title = ''     # Set title string to empty, this prevents printing of 'diagonals and' if only plotting coupling terms.
                else:

                    ax[0, dof - 3].plot(self.w / TwoPi, fowt.A_BEM[dof, dof2, :],  label=f'$A_{{{dof + 1},{dof2 + 1}}}$')
                    ax[1, dof - 3].plot(self.w / TwoPi, fowt.B_BEM[dof, dof2, :],  label=f'$B_{{{dof + 1},{dof2 + 1}}}$')
                    ax[0, dof - 3].legend(loc = 1)
                    ax[1, dof - 3].legend(loc = 1)

        ax[0, 0].set_title('Roll')
        ax[0, 1].set_title('Pitch')
        ax[0, 2].set_title('Yaw')
        ax[1, 0].set_xlabel('Frequency [Hz]')
        ax[1, 1].set_xlabel('Frequency [Hz]')
        ax[1, 2].set_xlabel('Frequency [Hz]')
        ax[0, 0].set_ylabel(f'$A_{{BEM}}$ $[kg m^2]$')
        ax[1, 0].set_ylabel(f'$B_{{BEM}}$ $[kg m^2/s]$')
        fig.suptitle(f'Roll, Pitch, Yaw BEM{diag_title} coupling terms.')

    def plotAeroTerms(self):
        # fowt = self.fowtList[0]
        metrics = self.results['case_metrics']
        variableXaxis = []
        DOF1 = [4, 6]
        DOF2 = [1, 6]

        if self.changeType in ['misalignment', 'floaterRotation', 'windSpeed', 'windMisalignment']:
            nCases = len(metrics['surge_avg'])
        else:
            nCases = 1
        for iCase in range(nCases):
            cases = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][iCase]))
            _, _, titleString = retrieveAxisParAnalysis(iCase, cases, self.changeType, variableXaxis,
                                                        self.design['parametricAnalysis'])
            fig, ax = plt.subplots(2, 3, sharex=True, figsize=get_figsize(self.latex_width, subplots=(2.5,3)))
            for dof in range(DOF1[0] - 1, DOF1[1]):
                for dof2 in range(DOF2[0] - 1, DOF2[1]):
                    if dof2 == 5:
                        ax[0, dof - 3].plot(self.w / TwoPi, self.fowt_A_aero_stored[iCase][dof, dof2, :],
                                            label=f'$A_{{{dof + 1},{dof2 + 1}}}$')
                        ax[1, dof - 3].plot(self.w / TwoPi, self.fowt_B_aero_stored[iCase][dof, dof2, :],
                                            label=f'$B_{{{dof + 1},{dof2 + 1}}}$')
                    else:
                        ax[0, dof - 3].plot(self.w / TwoPi, self.fowt_A_aero_stored[iCase][dof, dof2, :],
                                            label=f'$A_{{{dof + 1},{dof2 + 1}}}$')
                        ax[1, dof - 3].plot(self.w / TwoPi, self.fowt_B_aero_stored[iCase][dof, dof2, :],
                                            label=f'$B_{{{dof + 1},{dof2 + 1}}}$')
                    ax[0, dof - 3].legend(loc = 1)
                    ax[1, dof - 3].legend(loc = 1)

            ax[0, 0].set_title('Roll')
            ax[0, 1].set_title('Pitch')
            ax[0, 2].set_title('Yaw')
            ax[1, 0].set_xlabel('Frequency [Hz]')
            ax[1, 1].set_xlabel('Frequency [Hz]')
            ax[1, 2].set_xlabel('Frequency [Hz]')
            ax[0, 0].set_ylabel(f'$A_{{aero}}$ $[kg m^2]$')
            ax[1, 0].set_ylabel(f'$B_{{aero}}$ $[kg m^2/s]$')
            fig.suptitle(f'Roll, Pitch, Yaw Aerodynamic diagonal and coupling terms ({titleString}).')

    def plotCouplingContribution(self, diagonal = True):
        diag_title = ' diagonal and'

        # a loop could be added here for an array
        fowt = self.fowtList[0]
        M_A_contr = np.zeros_like(fowt.A_BEM, dtype=complex)
        B_contr = np.zeros_like(fowt.B_BEM, dtype=complex)
        C_contr = np.zeros_like(fowt.B_BEM, dtype=complex)
        variableXaxis= []
        metrics = self.results['case_metrics']
        nCases = len(metrics['surge_avg'])

        DOF1 = [4, 6]
        DOF2 = [1, 6]

        for iCase in range(nCases):
            cases = dict(zip(self.design['cases']['keys'], self.design['cases']['data'][iCase]))
            _, _, titleString = retrieveAxisParAnalysis(iCase, cases, self.changeType, variableXaxis,
                                                        self.design['parametricAnalysis'])
            fig, ax = plt.subplots(3, 3, sharex=True, figsize=get_figsize(self.latex_width, subplots=(3,3)))

            for dof in range(DOF1[0] - 1, DOF1[1]):
                for dof2 in range(DOF2[0] - 1, DOF2[1]):
                    if dof == dof2 and diagonal is not True:
                        diag_title = ''
                    else:

                        M_A_contr[dof, dof2, :] = np.multiply(-self.w ** 2,
                                                              np.multiply(self.M_tot_store[iCase][dof, dof2, :],
                                                                          self.Xi_store[iCase][dof2, :]))
                        B_contr[dof, dof2, :] = np.multiply(1j * self.w[:],
                                                            np.multiply(self.B_tot_store[iCase][dof, dof2, :],
                                                                        self.Xi_store[iCase][dof2, :]))
                        C_contr[dof, dof2, :] = np.multiply(self.C_tot_store[iCase][dof, dof2, :], self.Xi_store[iCase][dof2, :])

                        ax[0, dof - 3].plot(self.w / TwoPi, np.real(M_A_contr[dof, dof2, :]),
                                            label = f'$\\xi_{dof2 + 1}$') #label=f'$-\omega^2*(M+A)_{{{dof + 1},{dof2 + 1}}}*\\xi_{{{dof2 + 1}}}$')
                        ax[1, dof - 3].plot(self.w / TwoPi, np.real(B_contr[dof, dof2, :]),
                                            label = f'$\\xi_{dof2 + 1}$') #label=f'$j\omega*B_{{{dof + 1},{dof2 + 1}}}*\\xi_{dof2 + 1}$')
                        ax[2, dof - 3].plot(self.w / TwoPi, np.real(C_contr[dof, dof2, :]),
                                            label = f'$\\xi_{dof2 + 1}$') #label=f'$C_{{{dof + 1},{dof2 + 1}}}*\\xi_{dof2 + 1}$')
                        ax[0, dof - 3].legend(loc = 1)
                        ax[1, dof - 3].legend(loc = 1)
                        ax[2, dof - 3].legend(loc = 1)

            ax[0, 0].set_title('Roll')
            ax[0, 1].set_title('Pitch')
            ax[0, 2].set_title('Yaw')
            ax[2, 0].set_xlabel('Frequency [Hz]')
            ax[2, 1].set_xlabel('Frequency [Hz]')
            ax[2, 2].set_xlabel('Frequency [Hz]')
            ax[0, 0].set_ylabel(f'$-\omega^2*(M+A)*\\xi$ $[Nm]$')
            ax[1, 0].set_ylabel(f'$j\omega*B*\\xi$ $[Nm]$')
            ax[2, 0].set_ylabel(f'$C*\\xi$ $[Nm]$')
            fig.suptitle(f'Roll, Pitch, Yaw response contributions{diag_title} coupling terms ({titleString}).')

    def preprocess_HAMS(self, dw=0, wMax=0, dz=0, da=0):
        '''This generates a mesh for the platform, runs a BEM analysis on it
        using pyHAMS, and writes .1 and .3 output files for use with OpenFAST.
        The input parameters are useful for multifidelity applications where 
        different levels have different accuracy demands for the HAMS analysis.
        The mesh is only made for non-interesecting members flagged with potMod=1.
        
        PARAMETERS
        ----------
        dw : float
            Optional specification of custom frequency increment (rad/s).
        wMax : float
            Optional specification of maximum frequency for BEM analysis (rad/s). Will only be
            used if it is greater than the maximum frequency used in RAFT.
        dz : float
            desired longitudinal panel size for potential flow BEM analysis (m)
        da : float
            desired azimuthal panel size for potential flow BEM analysis (m)
        '''

        self.fowtList[0].calcBEM(dw=dw, wMax=wMax, dz=dz, da=da)

    def plot(self, ax=None, hideGrid=False, color='k', nodes=0):
        '''plots the whole model, including FOWTs and mooring system...'''

        # for now, start the plot via the mooring system, since MoorPy doesn't yet know how to draw on other codes' plots
        # self.ms.bodyList[0].setPosition(np.zeros(6))
        # self.ms.initialize()

        # fig = plt.figure(figsize=(20/2.54,12/2.54))
        # ax = Axes3D(fig)

        # if axes not passed in, make a new figure
        if ax == None:
            fig, ax = self.ms.plot(color=color)
        else:
            fig = ax.get_figure()
            self.ms.plot(ax=ax, color=color)

        # plot each FOWT
        for fowt in self.fowtList:
            fowt.plot(ax, color=color, nodes=nodes)

        if hideGrid:
            ax.set_xticks([])  # Hide axes ticks
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)  # Hide grid lines
            ax.grid(b=None)
            ax.axis('off')
            ax.set_frame_on(False)

        return fig, ax


def runRAFT(input_file, turbine_file=""):
    '''
    This will set up and run RAFT based on a YAML input file.
    '''

    # open the design YAML file and parse it into a dictionary for passing to raft
    print("Loading RAFT input file: " + input_file)

    with open(input_file) as file:
        design = yaml.load(file, Loader=yaml.FullLoader)

    print(f"'{design['name']}'")

    depth = float(design['mooring']['water_depth'])

    # for now, turn off potMod in the design dictionary to avoid BEM analysis
    # design['platform']['potModMaster'] = 1

    # read in turbine data and combine it in
    # if len(turbine_file) > 0:
    #   turbine = convertIEAturbineYAML2RAFT(turbine_file)
    #   design['turbine'].update(turbine)

    # Create and run the model
    print(" --- making model ---")
    model = raft.Model(design)
    print(" --- analyizing unloaded ---")
    model.analyzeUnloaded()
    print(" --- analyzing cases ---")
    model.analyzeCases()

    model.plot()

    model.plotResponses()

    # model.preprocess_HAMS("testHAMSoutput", dw=0.1, wMax=10)

    plt.show()

    return model


if __name__ == "__main__":
    import raft

    # model = runRAFT(os.path.join(raft_dir,'designs/DTU10MW.yaml'))
    model = runRAFT(os.path.join(raft_dir, 'designs/VolturnUS-S.yaml'))
    # model = runRAFT(os.path.join(raft_dir,'designs/OC3spar.yaml'))
    fowt = model.fowtList[0]
