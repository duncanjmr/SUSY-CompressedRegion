import matplotlib.pyplot as plt
import os
from susy_tools import *
import multiprocessing as mp
from scipy.optimize import curve_fit
from PIL import Image
import matplotlib
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
from scipy.interpolate import interp2d
import json

########################## Input Parameters ##########################
tanB = 50
m_sleptons = 450
N = (9, 12)
min_diff = 1.
max_diff = 70.
min_M2 = 120
max_M2 = 350
sign_M1 = 1.

N_checkmate = 21
to_optimize_gm2 = True

om_cm_bound_factor = 0.
dd_cm_bound_factor = 1.e10
show_plot = True

######################### Initial Setup ##################################

print()
print("Generating Checkmate Run Points for:")
print("\t tanB: %i" % tanB)
print("\t m_sleptons: %i" % m_sleptons)
print("\t M2 Limits: (%i, %i)" % (min_M2, max_M2))
print("\t Sign of M1: %i" % (sign_M1))
print()

## Edit susy-hit input to generate spectra
changeParamValue("A_t", 3.50e3)
changeParamValue("A_tau", -250)
changeParamValue("tanbeta(MZ)", tanB)

changeParamValue("M_eL",   m_sleptons)
changeParamValue("M_eR",   m_sleptons)
changeParamValue("M_muL",  m_sleptons)
changeParamValue("M_muR",  m_sleptons)
changeParamValue("M_tauL", m_sleptons)
changeParamValue("M_tauR", m_sleptons)

direc = "%i_%i_%i" % (tanB, m_sleptons, sign_M1)
if direc not in os.listdir("./"):
    os.mkdir(direc)
    
M2_init = np.logspace(np.log10(min_M2), np.log10(max_M2), N[0])

print("Results will be saved in ./%s" % direc)
    
################ Scan Point Generation #######################

points = []
points_l = []
mu = []

M2_l = []

M1_max = []
diffs = []

print("Calculating minima of delta(x1,x2), and fixing mu with g-2, for each M2: \n")
print(" M2 \t  M1 \t mu")
print("------------------------")

for i, M2 in enumerate(M2_init):
    
    print("%i: " % M2, end="")
        
        
    # Set mu with g-2 optimization, if requested
    if to_optimize_gm2:
        m = optimize_gm2(sign_M1*M2, M2, M2+1, tanB, m_sleptons, to_minimize="mu")
        if np.isnan(m):
            print("\t NaN \t NaN")
            continue
        mu.append(m)
        changeParamValue("mu(EWSB)", m)
        M2_l.append(M2)
        
    else:
        changeParamValue("mu(EWSB)", M2)
        mu.append(M2)
        M2_l.append(M2)
    
    # Now find the mass splitting minimum.
    M_min, diff = minimize_neutralinoMassDiff(sign_M1*M2, M2, "M1", step=0.5, return_diff=True,verbose=False)
    M1_m = M_min[0]
    M1_m -= 0.1 * np.sign(M1_m)
    
    M1_max.append(M1_m)
    diffs.append(diff)
    print("\t %i \t %i" % (M1_m, m))
    
# This generates a y-axis sequence of points to scan
print()
mu_l = []
M1_max = np.abs(M1_max) - min_diff

# Use fewer points if the mass splitting minimum is large
start = (np.array(diffs) + min_diff) / max_diff
N_adj = np.round(N[1] * np.sqrt(np.log10(start) / np.log10(np.min(start))), 0 )
    
points_l = []
for i in range(len(M1_max)):
    # Generate points along the y axis, even in logspace
    points = []
    for dm in (np.logspace(np.log10(start[i]), 0, int(N_adj[i])) - start[i] + 1./max_diff)/(1-start[i]):
        points.append([M1_max[i]* (1-dm*max_diff/M1_max[i]), M2_l[i]])

    if sign_M1 < 0:
        points = [[-p[0], p[1]] for p in points]
    points_l.append(points)  
    mu_l.extend([mu[i]] * int(N_adj[i]))
    
def additional_command(M1, M2, index):
    changeParamValue("mu(EWSB)", mu_l[index])
    
# Run susyhit and micromegas
data = run(np.vstack(points_l), True, run_prospino=False, 
               run_micromegas=True,
               run_checkmate=False,
               working_directory="./%s/scan" % (direc),
               additional_command=additional_command)
        
    
data["tanB"] = tanB
data["m_sleptons"] = m_sleptons
data["sign_M1"] = sign_M1
data["mu"] = mu_l

with open("%s/scan_points.json" % direc, "w") as outfile:
    json.dump(data, outfile)

    
###############

br_gam = []
points_new = []

# Extract the photon branching ratio for plot showing 
for i in range(len(data["M2"])):

    filename = "spectrum_%i_%i.dat" % (data["M1"][i], data["M2"][i])
    
    with open("./%s/scan/spectra_slha/" % direc + filename) as f:
        s = f.read()

    points_new.append([data["M1"][i], data["M2"][i]])
    br_gam.append(np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10 gam)")))


print()

gm2 = data["gm2"]
om = data["omega_dm"]
dd = data["dd_pval"]

############################## Checkmate point Generation #########################

print()
print("Scanning Complete. Generating %i Checkmate points in allowed region: " % N_checkmate)
x = np.array(data["M2"])
y = np.abs(data["~chi_20"]) - np.abs(data["~chi_10"])

sel = ~np.isnan(y) * ~np.isnan(x)
checkmate_points, boundary = get_checkmatePoints(x[sel], np.log(y[sel]), om_cm_bound_factor*np.array(om)[sel], 
                                                 dd_cm_bound_factor*(np.array(dd)[sel]+1e-5), N=N_checkmate)

boundary[:,1] = np.exp(boundary[:,1])
checkmate_points[:,1] = np.exp(checkmate_points[:,1])
mu_cm = np.interp(checkmate_points.T[0], M2_l, mu)

p0 = np.array(points_new)[sel]
tck = interp.bisplrep(x, y, p0.T[0])
M1_cm = np.array([interp.bisplev(*c, tck) for c in checkmate_points])

tck = interp.bisplrep(x, y, p0.T[1])
M2_cm = np.array([interp.bisplev(*c, tck) for c in checkmate_points])

points = np.vstack((M1_cm, M2_cm)).T

import json
      
# Data to be written 
dictionary ={ 
  "points": [list(c) for c in points],
  "mu": list(mu_cm),
  "tanB": tanB,
  "m_sleptons": m_sleptons,
  "sign_M1": sign_M1
} 
      
# Serializing json  
cm_file = "%s/checkmate_points.json" % direc
with open(cm_file, "w") as outfile:
    json.dump(dictionary, outfile)
    
print("Generation Complete. Saved to json named %s" % cm_file)
print()

########### Run susyhit/micromegas for initial checkmate points #########


print("Running susyhit/micromegas for checkmate points")

def additional_command(M1, M2, index):
    changeParamValue("mu(EWSB)", mu_cm[index])

data = run(np.vstack(points), True, run_prospino=False, 
               run_micromegas=True,
               run_checkmate=False,
               working_directory="./%s/checkmate" % (direc),
               additional_command=additional_command)

print()
######################## Plot points and show ###########################

f = plt.figure(figsize=(10, 5))
ax = f.add_subplot(1, 2, 1)

show_omega=True

name = "$\chi_1^0 + \gamma$"
levels = np.linspace(0,1, 11)

br = np.array(br_gam)
select = ~np.isnan(br) * sel

xsel = x[select]
ysel = y[select]
br_nonan = br[select]

mx = max(br_nonan)
if np.all(br_nonan == 0):
    mn = 0
else:
    mn = 0.01*min(br_nonan[br_nonan != 0])

cmap_br = truncate_colormap(plt.get_cmap("Reds"), mn, np.min([mx, 1.]))

cs = ax.tricontourf(xsel, ysel, br_nonan, levels=levels, 
                alpha=0.5, cmap=cmap_br)

# Plot contour lines
cmapblack = truncate_colormap(plt.get_cmap("Greys"), 0.9, 1)
cs = ax.tricontour(xsel, ysel, br_nonan, levels=levels,
                  cmap=cmapblack, linewidths=1)

ax.clabel(cs, cs.levels, inline=True, fontsize=10)

# Set some plot labels
ax.set_xlabel(r"M2 ($\mu$ fixed to g-2) [GeV]", fontsize=12)
ax.set_title(r"BR$(\chi_2^0\  \to $ %s)" % name, fontsize=16)
ax.set_yscale("log")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(1)

# Plot omega_dm bounds
s2 = ~np.isnan(om)[sel]
ax.tricontourf(x[s2], y[s2], np.array(om)[sel][s2], levels=[0.07, 0.3], cmap="Purples", alpha=0.8)

# Plot direct detection exclusion
plt.tricontourf(x, y, np.array(dd)[sel], levels = [0, 0.05], cmap="Greys",alpha=0.7)

ax.set_ylabel(r"$\Delta(\chi_1^0, \chi_2^0)$ [GeV]", fontsize=12)

# Plot checkmate points
plt.scatter(x, y, marker="x", alpha=0.8)
plt.plot(*boundary.T)
cm_plot = plt.scatter(*checkmate_points.T, marker="x", alpha=1, c="red")

# Plot mu vs M2 
ax = f.add_subplot(1, 2, 2)
ax.plot(M2_l, mu)
ax.set_xscale("log")
ax.set_title("M2 vs. mu", fontsize=15)
ax.set_xlabel("M2 [GeV]", fontsize=12)
ax.set_ylabel("mu [GeV]", fontsize=12)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, which="both")

sign = "\nM1<0"
plt.text(np.max(x)*0.75, ax.get_ylim()[1]*0.9, 
         "tanB: %i\nm_sl: %i GeV"% (tanB, m_sleptons) + sign*int(sign_M1 < 0), 
         fontsize=12, bbox=dict(facecolor='white', edgecolor='black',))
ax.legend([Patch(facecolor='purple', label='Color Patch', alpha=0.4),
           Patch(facecolor='grey', label='Color Patch', alpha=0.4),
           cm_plot
          ],
          [r"$\Omega = 0.07 - 0.3$",
           r"DD p-value < 0.05",
           r"Checkmate Run Points",], fontsize=12, loc=3)


plt.tight_layout()


# Save, open image, and then remove it
if show_plot:
    plt.savefig("%s/generated_points.png" % direc)

    img = Image.open('%s/generated_points.png' % direc)
    img.show() 

                                          
