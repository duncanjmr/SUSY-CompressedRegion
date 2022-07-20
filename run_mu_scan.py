
import numpy as np
import os
import json
from susy_tools import *

save_for_cluster = True

tanB = 50
m_sleptons = 500
sign_M1 = -1

save_directory = "cluster_muM2scan_%i_%i_%i" % (tanB, m_sleptons, sign_M1)


N=(2,2)

M2_l = np.linspace(150, m_sleptons, N[0])
max_mu = 0.3e4

min_mu = M2_l[0] - 10

p = []

for m2 in M2_l:
    mn = m2 - 10
    N_adj = int(np.ceil(np.log(max_mu/m2) / np.log(max_mu/min_mu) * N[1]))

    for mu in np.logspace(np.log10(mn), np.log10(max_mu), N_adj):
        p.append([m2, mu])

M1_l = []
i=1
i2=1
print("%i/%i\r" % (i, len(p)), end="")

for M2, mu in p:
    
    m1, d = optimize_relicDensity(sign_M1*M2, M2, mu, m_sleptons, tanB)

    if not np.isnan(m1):
        d2 = run([[m1,M2]], True, verbose=False)
        if i2 == 1:
            dic = d
            dic["mu"] = [mu]
            dic["dd_pval"] = d2["dd_pval"]
        else:
            for k in d.keys():
                dic[k].append(d[k][0])
            dic["mu"].append(mu)
            dic["dd_pval"].extend(d2["dd_pval"])
            
        print("%i/%i, last (mu, M2, omega, dd_pval): (%i, %i, %.2f, %.2f) \r" % (i, len(p), mu, M2, dic["omega_dm"][-1], dic["dd_pval"][-1]), end="")
        i2 += 1


    #print(dic["mu"], dic["M2"])
    i += 1
    
print()

with open("grid_mu_scan_%i_%i_%i.json" % (m_sleptons, tanB, sign_M1), "w") as f:
    json.dump(dic, f)
    
if save_for_cluster == True:
    
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    for subdir in ["spectra_slha", "prospino_cx", "checkmate", "micromegas_out"]:
        if subdir not in os.listdir(save_directory):
            os.mkdir(save_directory + "/" + subdir) 

    for t in dic["tag"]:
        os.system("cp test/spectra_slha/spectrum_%s.dat %s/spectra_slha/" % (t, save_directory))
        
    cl_dic = {}
    cl_dic["index"] = [a for a in range(len(dic["mu"]))]
    cl_dic["directory"] = [save_directory for i in range(len(dic["mu"]))]
    cl_dic["filename"] = ["spectrum_%s.dat" % t for t in dic["tag"]]
    
    with open("cluster_index_reference.txt", "w") as f:
        json.dump(cl_dic, f)