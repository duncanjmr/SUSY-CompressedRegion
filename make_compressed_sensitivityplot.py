import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import matplotlib.patches as mpatches
from susy_tools import *
from scipy.interpolate import interp2d
import scipy.interpolate as interp
import multiprocessing as mp
import time
import queue
import datetime
from scipy.optimize import minimize 


font = {'size'   : 14}
plt.rc('font', **font)

############## Specify list of points to test ##############
####### M1 goes in the first column, M2 in the second ######

max_M1 = 400
min_M1 = 50
max_diff = 80.
N = 5
min_diff = 1.

points = []
for x in np.linspace(min_M1, max_M1, N):
    min_M2 = get_closestmass(x)
    mindiff = get_neutralinoMassDiff(x, min_M2)
    
    points.append([x, min_M2])
    for y in np.logspace(np.log10(mindiff), np.log10(max_diff), N-1):
        points.append([x, min_M2 + y])
        
mu = 2000 # Specify MSSM parameters
tanB = 15
m_sleptons = 300

############################################################
##################### Input Parameters #####################
############################################################

# Select the region to plot
# options: "full", "compressed"
# Compressed only plots the region m_x2 - m_x1 < m_higgs
region = "compressed"

remake = False #{"susyhit": False, "prospino": False, "micromegas": True} # Set to true to recalculate all files
save_cx = True # Saves the generated g-2 contributions to a file

# Choose which programs to use
use_prospino = True # For cross sections
use_micromegas = True # For g-2, omega

# Optionally choose which limits to draw on the plot output
draw_collider_limits = True
draw_gm2_limits = True
draw_omega_limits = False

show_ul_points = True
show_pointlabels = False

# Give the measured g-2 anomaly, and uncertainty.
gm2_measured = 2.51e-9
gm2_sigma = 0.59e-9

working_directory = "cx_%i_%i_%i" % (mu, tanB, m_sleptons)
#f(optimize_higgs())

############### Plotting Settings ##########################

use_lumi = 139.

files = ["./HEPData-ins1771533-v2-Upper_Limits_3l.csv",
         "./HEPData-ins1658902-X1X2-toleptons2.csv",
         "./HEPData-ins1767649-compressed.csv"]

is_compressed = [False, False, True]

#lumi = [36.1, 36.1, 139., 139., 36.1]
lumi = [36.1, 139., 139.]
unit_conversion = [1., 0.001, 1.]

labels = ["3lbb", "2/3l, ~l-mediated", "compressed ULs"]

show = [False, False, True]

cmaps = [truncate_colormap(plt.get_cmap("gray"), 0.2, 0.3), 
         truncate_colormap(plt.get_cmap("spring"), 0, 0.1),
         truncate_colormap(plt.get_cmap("cool"), 0.2, 0.3),
         truncate_colormap(plt.get_cmap("YlGn"), 0.3, 0.4),
         truncate_colormap(plt.get_cmap("Oranges"), 0.2, 0.3),]

alphas = [0.4, 0.4, 0.4, 0.4, 0.4]

gm2_cmap = truncate_colormap(plt.get_cmap("YlGn"), 0.3, 0.7)
gm2_alpha= 0.25


############################################################
############### Calculating cross sections #################
############################################################

## Edit susy-hit input to generate spectra
changeParamValue("mu(EWSB)", mu)
changeParamValue("A_t", 3.50e3)
changeParamValue("A_tau", -250)
changeParamValue("tanbeta(MZ)", tanB)

changeParamValue("M_eL",   m_sleptons)
changeParamValue("M_eR",   m_sleptons)
changeParamValue("M_muL",  m_sleptons)
changeParamValue("M_muR",  m_sleptons)
changeParamValue("M_tauL", m_sleptons)
changeParamValue("M_tauR", m_sleptons)


print("Starting calculations for specified points...\n")
print("Saving to: %s \n", working_directory)
print("Settings: ")
print("\t Run prospino: %s" % str(use_prospino))
print("\t Run micromegas: %s\n" % str(use_micromegas))
print("Creating %i Processes..." % len(points))

queue = run(points, remake, use_prospino, use_micromegas, working_directory)
data = []
while queue.qsize() > 0:
    ret = queue.get()
    data.append([ret[i] for i in ["m_x1", "m_x2", "cx", "gm2", "omega"]])
    
m_x1, m_x2, cx, gm2, om = np.array(data).T  
m_x1 = np.abs(m_x1)
m_x2 = np.abs(m_x2)

ymax = np.max(m_x1[~np.isnan(m_x1)])
xmax = np.max(m_x2[~np.isnan(m_x2)])

print()
print("Done!")

#print("Max, Min of higgs mass: %.1f, %.1f" %(np.max(h[~np.isnan(h)]), np.min(h[~np.isnan(h)])))

################## Plot Making ###############################

plt.figure(figsize=(10,8))
runs = plt.scatter(m_x2, np.abs(m_x2 - m_x1), c="C1")

M1, M2 = np.array(points).T
#plt.scatter(M2, np.abs(M2-M1))
if show_pointlabels:
    for (x2, x1, c, g) in zip(m_x2, m_x1, cx, gm2):
        if ~np.isnan(x2) and ~np.isnan(x1):
            plt.text(x2, (x2-x1), "%.1f, %.1E" % (c, g), fontsize=11)

if "temp.out" in os.listdir("."):
    os.system("rm temp.out")
    
######################## Draw g-2 limits ####################

sel_gm2 = ~np.isnan(m_x2) * ~np.isnan(m_x1) * ~np.isnan(gm2)
if use_micromegas and draw_gm2_limits and sum(sel_gm2) > 5:

    levels = [0, 1]

    validity = -(gm2[sel_gm2] - gm2_measured)**2 + gm2_sigma**2

    if np.max(gm2[sel_gm2]) < gm2_measured - gm2_sigma:
        print("g-2 contribution is too small!")
        print("Largest: %.1E\t Threshold: %.1E" % (np.max(gm2[sel_gm2]), 
                                                   gm2_measured - gm2_sigma))
    elif np.min(gm2[sel_gm2]) > gm2_measured + gm2_sigma:
        print("g-2 contribution is too large!")
        print("Smallest: %.1E\t Threshold: %.1E" % (np.min(gm2[sel_gm2]), 
                                                    gm2_measured + gm2_sigma))

      
    cont_lims = [gm2_measured - gm2_sigma, gm2_measured + gm2_sigma]
    levels = np.linspace(*cont_lims, 10)
    
    if np.any(validity > 0):
        f = plt.tricontourf(m_x2[sel_gm2], (m_x2 - m_x1)[sel_gm2], gm2[sel_gm2], 
                            levels = levels, cmap=gm2_cmap, alpha=gm2_alpha)
        plt.tricontour(m_x2[sel_gm2], (m_x2 - m_x1)[sel_gm2], gm2[sel_gm2], 
                       levels = cont_lims, cmap=gm2_cmap, alpha=gm2_alpha+0.2)
        
        plt.colorbar(f)

##############################################################


sel = ~np.isnan(m_x2) * ~np.isnan(m_x1) * ~np.isnan(cx)

def isfloat(inp):
    try:
        float(inp)
        return True
    except: return False
    

if use_prospino and draw_collider_limits and sum(sel) > 3:
    s = sum(sel) #-np.sqrt(2*sum(sel))
    #print(m_x1, m_x2)
    print("s =", s)
    #tck = interp.bisplrep(m_x2[sel], m_x1[sel], np.log(cx[sel]), s=s/2)

    data = np.array([m_x1, m_x2, cx]).T[sel]
    colors = []
    for i in range(len(files)):
        
        if not show[i]:
            continue

        ## Load the ppper limit files, trim down to the relevant data
        with open(files[i]) as f:
            s = f.read()
            lines = s.split("\n")
            while not isfloat(lines[0].split(",")[0]):
                lines = lines[1:]
            
            while lines[-1] == "":
                lines = lines[:-1]
                
            file = np.array([line.split(",") for line in lines], dtype=float)
            
        if not is_compressed[i]:
            file[:, 0] = file[:, 1] - file[:,0]
        
        data[:, 0] = data[:, 1] - data[:,0]
            
        draw_contour(data, file, lumi[i], unit_conversion[i], cmap=cmaps[i], 
                     show_ul_points=show_ul_points, 
                     show_pointlabels=show_pointlabels, alpha=0.3)
        
            
        colors.append(cmaps[i](0))
else:
    colors = []
    show = [False] * len(files)
        
if draw_gm2_limits:
    colors.append(gm2_cmap(0))
    labels.append(r"$\Delta g_{muon}$")

plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.tick_params(bottom=True, top=True, left=True, right=True)

# Finishing touches
plt.xlabel(r"$m_{\chi_0^2}$")
plt.ylabel(r"$\Delta(m_{\chi_0^1}, m_{\chi_0^2})$")
plt.title(r"Experimental Constraints on $\chi_0^2 \ \chi_1^\pm$ production")

plt.xlim(92, xmax)

plt.ylim(min_diff, max_diff*2)
plt.yscale("log")

plt.text((plt.xlim()[1]-92)*0.7 + 92, 0.01*plt.ylim()[1], 
         r"$\mu$ = %i" % mu + "\n" + r"$\tan\beta = %i$" % tanB + "\n" +
         r"$m_{sleptons}$ = %i" % m_sleptons,
         bbox=dict(facecolor='white', edgecolor='black', pad=10.0) )

patches = [mpatches.Patch(color=colors[i], alpha=0.6) for i in range(len(colors))]

print(patches, labels)

plt.legend(patches, np.array(labels)[np.array(show + [True])])

plt.savefig(working_directory + "/cx_sensitivity.png")
plt.savefig("./cx_sensitivity_recent.png")


if save_cx:
    data = np.vstack((m_x1, m_x2, cx)).T
    if "cx_list.csv" in os.listdir(working_directory):
        os.system("rm " + working_directory + "/cx_list.csv")
    np.savetxt(working_directory + "/cx_list.csv", data, delimiter=",")
    
os.system("cp ./draw_cx_bounds_useAll.py " + working_directory)
os.system("cp ./susy_tools.py " + working_directory)




