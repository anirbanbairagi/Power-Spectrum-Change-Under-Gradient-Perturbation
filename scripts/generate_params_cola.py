import os
import numpy as np

rootDir = "/u/nchartier/PkCorr/COLA_source/"           # parameter files + shared redshift file
filePrefix = "Cola"
basePrefix = "Cola"
snapOutDir = "/work/hdd/bdne/nchartier/cola_lh/"
cosmoDir = "/work/hdd/bdne/nchartier/quijote_lh"       # per-realization Cosmo_params.dat + Pk_mm_z=0.000.txt

nRealizations = 50                                     # LH realizations 0..49
timeStepsList = ["20"]

#%% Parameters for the L-Picola source files (cosmology/seed/paths filled in per-realization below)

paramDictionary = {"OutputDir":snapOutDir, "FileBase":"", "OutputRedshiftFile":"", "NumFilesWrittenInParallel":"8", "UseCOLA": "1", "Buffer":"1.3", "Nmesh":"512", "Nsample":"512"
                   , "Box":"1000000.0", "Init_Redshift":"127.0", "Seed":"", "SphereMode":"0", "WhichSpectrum":"1", "WhichTransfer":"0", "FileWithInputTransfer":"IDC",
                   "FileWithInputSpectrum":"", "Omega":"", "OmegaBaryon":"", "OmegaLambda":"", "HubbleParam":"", "Sigma8":""
                   , "PrimordialIndex":"", "StepDist":"0", "DeltaA":"0", "nLPT":"0.5", "UnitLength_in_cm":"3.085678e21", "UnitMass_in_g":"1.989e43",
                   "UnitVelocity_in_cm_per_s":"1e5", "InputSpectrum_UnitLength_in_cm":"3.085678e24"}

redshiftDict = {"20":"redshifts_20steps_from127.dat", "40":"redshifts_40steps_from127.dat", "ICE":"redshifts_ICEsteps_from127.dat"}
nLPTStr = "pos0p5"

#%% WRITING EACH PARAMETER FILE

for k in range(nRealizations):

    Omega_m, Omega_b, h, ns, sigma8 = np.loadtxt(os.path.join(cosmoDir, str(k), "Cosmo_params.dat"))

    paramDictionary["Seed"] = str(k)
    paramDictionary["Omega"] = str(Omega_m)
    paramDictionary["OmegaBaryon"] = str(Omega_b)
    paramDictionary["OmegaLambda"] = str(1.0 - Omega_m)
    paramDictionary["HubbleParam"] = str(h)
    paramDictionary["Sigma8"] = str(sigma8)
    paramDictionary["PrimordialIndex"] = str(ns)
    paramDictionary["FileWithInputSpectrum"] = os.path.join(cosmoDir, str(k), "Pk_mm_z=0.000.txt")
    paramDictionary["OutputDir"] = snapOutDir + str(k) + "/"
    os.makedirs(paramDictionary["OutputDir"], exist_ok=True)

    for n in range(len(timeStepsList)):

        fullPath = rootDir + str(k) + "/"
        fileName = filePrefix + str(k) + "_" + paramDictionary["Nmesh"] + "_" + timeStepsList[n] + "steps" + "_nLPT" + nLPTStr + ".dat"

        paramDictionary["FileBase"] = basePrefix + str(k) + "_" + paramDictionary["Nmesh"] + "_" + timeStepsList[n] + "steps" + "nLPT" + nLPTStr
        paramDictionary["OutputRedshiftFile"] = os.path.join(rootDir, redshiftDict[timeStepsList[n]])

        if not os.path.exists(fullPath):
            os.mkdir(fullPath)

        with open(os.path.join(fullPath, fileName), "w") as paramFile:
            for key, value in paramDictionary.items():
                paramFile.write("%s  %s\n" % (key, value))
