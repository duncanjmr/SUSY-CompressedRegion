import numpy as np
import os
from susy_tools import *
import multiprocessing as mp

# Input parameters
tanB = 50
M2 = 150.
N = (5, 5)
max_diff = 80.
max_mu = 2001.
optimize_gm2 = True
run_checkmate = False

changeParamValue("tanbeta(MZ)", tanB)

def changeSleptonMasses(m_s):
    #print(m_in[index])
    changeParamValue("M_eL",   m_s)
    changeParamValue("M_eR",   m_s)
    changeParamValue("M_muL",  m_s)
    changeParamValue("M_muR",  m_s)
    changeParamValue("M_tauL", m_s)
    changeParamValue("M_tauR", m_s)

# Slepton/higgsino scan points
mu = np.logspace(np.log10(M2+10), np.log10(max_mu), N[0])  


# Generate slepton masses and points to run over
m_sleptons = []
M1_max = []
diffs = []
for i, m in enumerate(mu):
    changeParamValue("mu(EWSB)", m)
    
    if optimize_gm2:
        m_sleptons.append(optimize_sleptonMassesGm2(M2, M2, m, tanB))
    else:
        changeSleptonMasses(m)
        m_sleptons.append(m)
        
    # Get the M1 that minimizes the neutralino mass difference, add to list
    M_min, diff = minimize_neutralinoMassDiff(M2, M2, "M1", return_diff=True)
    M1_m = M_min[0]
    M1_m -= 0.1 * np.sign(M1_m)
    
    M1_max.append(M1_m)
    diffs.append(diff)

    print("mu: %i    \t m_sleptons: %i" % (m, m_sleptons[i]))

# Generate points to run over
M1_min = M1_max[np.argmin(diffs)] - max_diff
start = (np.max(M1_max)-np.array(M1_max)) / max_diff + 0.001
N_adj = np.round(np.sqrt(np.log10(start) / np.min(np.log10(start))) * N[1], 0)

points_l = []
for i in range(len(M1_max)):
    points = []
    for dm in np.logspace(np.log10(start[i]), 0, int(N_adj[i])):
        points.append([np.max(M1_max) * (1-dm*max_diff/max(M1_max)), M2])
    points_l.append(points)  


# Finally, run the scan
for i, m in enumerate(mu):
    changeParamValue("mu(EWSB)", m)
    changeSleptonMasses(m_sleptons[i])
    
    data = run(points_l[i], remake=True, run_prospino=False, run_micromegas=True, 
               run_checkmate=run_checkmate,
               working_directory="scan_dir/mu_%i" % m)
    
    print()
