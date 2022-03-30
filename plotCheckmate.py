from susy_tools import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from matplotlib.patches import Patch
from PIL import Image
import argparse


# Parse filename arg
parser = argparse.ArgumentParser()
parser.add_argument("scan_dir", metavar="scan_dir", type=str)
args = vars(parser.parse_args())
direc = args["scan_dir"]

print("Plotting Checkmate results in %s" % direc)

with open("%s/checkmate_complete.json" % direc, "r") as f:
    cm_data = json.load(f)
    
with open("%s/scan_points.json" % direc, "r") as f:
    scan_data = json.load(f)

show_plot = True
smooth_checkmate = False

tanB = cm_data["tanB"]
m_sleptons = cm_data["m_sleptons"]
sign_M1 = cm_data["sign_M1"]

br_Z = []
br_h = []
br_stau = []
br_emu = []
br_snu = []
br_nu = []
br_gam = []
br_x0_quarks = []
br_xpm_quarks = []
br_sleptons = []
br_stau = []
br_charg = []
h_m = []

frac_wino = []
frac_bino = []

points_new = []

for i in range(len(scan_data["M2"])):

    filename = "spectrum_%i_%i.dat" % (scan_data["M1"][i], scan_data["M2"][i])

    with open("./%s/scan/spectra_slha/" % direc + filename) as f:
        s = f.read()

    points_new.append([scan_data["M1"][i], scan_data["M2"][i]])

    br_Z.append(np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10   Z )")))
    br_h.append(np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10   h )")))

    br_sleptons.append(np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~mu_L-    mu+)")) + 
                  np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~mu_L+    mu-)")) +
                  np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~e_L-     e+)")) +
                  np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~e_L+     e-)")) +
                  np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~tau_1+   tau-)")) +
                  np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~tau_1-   tau+)") ))

    br_emu.append(np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10 e+      e-)")) + 
                  np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10 mu+     mu-)")) )

    br_stau.append(np.nan_to_num(np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10 tau+    tau-)"))))

    br_snu.append(2 * getBranchingRatio(s, "BR(~chi_20 -> ~nu_eL*   nu_e ") + 
                  2 * getBranchingRatio(s, "BR(~chi_20 -> ~nu_muL*  nu_mu ") + 
                  2 * getBranchingRatio(s, "BR(~chi_20 -> ~nu_tau1* nu_tau "))

    br_nu.append(np.nan_to_num(3*getBranchingRatio(s, "BR(~chi_20 -> ~chi_10 nu_eb   nu_e)")))

    br_x0_quarks.append(np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10 ub      u)")) +
                     np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10 db      d)")) +
                     np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10 cb      c)")) +
                     np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10 sb      s)")) +
                     np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10 bb      b)")) 
                    )

    br_gam.append(np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10 gam)")))

    br_xpm_quarks.append(np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_1+ ub      d)")) +
                         np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_1- db      u)")) +
                         np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_1+ cb      s)")) +
                         np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_1- sb      c)"))
                        )

    br_charg.append(np.nan_to_num(getBranchingRatio(s, "BR(~chi_1+ -> ~chi_10 u    db)")) + 
                    np.nan_to_num(getBranchingRatio(s, "BR(~chi_1+ -> ~chi_10 c    sb)")))
    h_m.append(getParamValue(s, "h"))


    frac_bino.append(getParamValue(s, "N_11"))
    frac_wino.append(getParamValue(s, "N_12"))

print()

m_x1 = np.array(scan_data["~chi_10"])
m_x2 = np.array(scan_data["~chi_20"])
gm2 = np.array(scan_data["gm2"])
om = np.array(scan_data["omega_dm"])
dd = np.array(scan_data["dd_pval"])

cm_x = np.array(cm_data["M2"])
cm_y = np.abs(cm_data["~chi_20"]) - np.abs(cm_data["~chi_10"])

cm_x0 = cm_x
cm_y0 = cm_y
cm_r = np.array(cm_data["r"])
cm_r0 = cm_r
analyses = np.array(cm_data["analysis"])

m_x1 = np.array(m_x1)
m_x2 = np.array(m_x2)

y = np.abs(m_x2) - np.abs(m_x1)
x = np.array(scan_data["M2"])

sel = ~np.isnan(y) * ~np.isnan(x)

x = x[sel]
y = y[sel]

if smooth_checkmate:
    print("Smoothing checkmate points for more reasonable contours...")

    #print(np.vstack((cm_x, cm_y, cm_r)).T)
    interp_points, _ = get_checkmatePoints(cm_x, np.log(cm_y), np.zeros(len(cm_x)), np.ones(len(cm_y)), 50)
    interp_points[:, 1] = np.exp(interp_points[:, 1])
    
    tck = interp.bisplrep(cm_x, cm_y, np.log(cm_r), kx=2, ky=2, s=1)
    cm_r = np.array([np.exp(interp.bisplev(*p, tck)) for p in interp_points])
    cm_x, cm_y = interp_points.T
    
    #print(np.vstack((cm_x, cm_y, cm_r)).T)
    
print("Done.")
levels = np.linspace(0, 1.001, 100)

f = plt.figure(figsize=(24, 13))

show_omega=True

br_rat = [br_gam, br_x0_quarks, br_emu, br_stau, br_nu, br_charg, cm_r] #, br_nu]
names = [ "$\chi_1^0 + \gamma$", 
         "$\chi_1^0 + \overline{q} + q$", 
         "$\chi_1^0 + e/\mu^+ + e/\mu^-$", 
         r"$\chi_1^0 + \tau^+ + \tau^-$",
         "neutrinos", "chargino decay",
        "checkmate r-value"]

n_levels = 10
levels_l = [np.linspace(0,1, n_levels+1),
            np.linspace(0,1, n_levels+1),
            np.linspace(0, (np.max(br_emu) // 0.02+1) * 0.02, n_levels+1),
            np.linspace(0, 1, n_levels+1),
            np.linspace(0, np.max(br_nu) // 0.02 * 0.02+0.02, n_levels+1),
            np.linspace(0.4, 1, n_levels+1),
            np.linspace(0,1,n_levels+1)]

max_br = np.max(np.array(br_rat[:-1])[~np.isnan(br_rat[:-1])])
for i in range(len(br_rat)):

    ax = f.add_subplot(2, 4, i+1)
    
    br = np.array(br_rat[i])
    select = ~np.isnan(br)
    br_nonan = br[select]

    if i == 6:
        xsel = cm_x[select]
        ysel = cm_y[select]
    else:
        xsel = x[select*sel]
        ysel = y[select*sel]
    
    
    N_levels = 7
    
    mx = max(br_nonan)
    if np.all(br_nonan == 0):
        mn = 0
    else:
        mn = 0.01*min(br_nonan[br_nonan != 0])
            
    #levels = (1.015*mx - (mx-mn) * np.logspace(-2, 0, N_levels))[::-1]
    levels = levels_l[i]
    
    cmap_br = truncate_colormap(plt.get_cmap("Reds"), mn, np.min([mx, 1.]))
    
    cs = ax.tricontourf(xsel, ysel, br_nonan, levels=levels, 
                    alpha=0.5, cmap=cmap_br)
    

    cmapblack = truncate_colormap(plt.get_cmap("Greys"), 0.9, 1)
    cs = ax.tricontour(xsel, ysel, br_nonan, levels=levels,
                      cmap=cmapblack, linewidths=1)

    ax.clabel(cs, cs.levels, inline=True, fontsize=10)
    
    
    #cmap = truncate_colormap(plt.get_cmap("Blues"), 0.4, 0.55)
    #lim = ax.tricontour(x, y, np.abs(np.vstack(points_new)[sel].T[0]) - np.abs(np.vstack(points_l)[sel].T[1]), 
    #                    levels= [-M2, 0], cmap=cmap)

    ax.set_xlabel(r"M2 ($\mu$ fixed to g-2) [GeV]", fontsize=12)
    ax.set_title(r"BR$(\chi_2^0\  \to $ %s)" % names[i], fontsize=16)
    ax.set_yscale("log")
    #ax.set_xscale("log")

    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())

    ax.tick_params(axis="x", which="both", rotation=45)
    #for tick in ax.get_xticklabels():
    #    tick.set_rotation(45)
    
    ax.grid(1)
    
    if np.any(~np.isnan(cm_r)):
        s2 = ~np.isnan(cm_r)
        plt.tricontour(cm_x[s2], cm_y[s2], cm_r[s2],levels=[1,1e2], cmap="autumn", linestyles="dashed")
        plt.tricontourf(cm_x[s2], cm_y[s2], cm_r[s2],levels=[1,1e2], cmap="autumn", alpha=0.3)
    
    if show_omega:
        
        s2 = ~np.isnan(om)[sel]
        ax.tricontourf(x[s2], y[s2], np.array(om)[sel][s2], levels=[0.07, 0.3], cmap="Purples", alpha=0.8)
        cs = ax.tricontour(x[s2], y[s2], np.array(om)[sel][s2], levels=[0.07, 0.3], cmap="Purples", alpha=0)
        dd_cont = plt.tricontourf(x, y, np.array(dd)[sel], levels = [0, 0.05], cmap="Greys",alpha=0.7)
        
    
    if i == 1:
        ax.legend([Patch(facecolor='purple', label='Color Patch', alpha=0.4),
                   Patch(facecolor='grey', label='Color Patch', alpha=0.4),
                   Patch(facecolor='Orange', label='Color Patch', alpha=1)
                  ],
                  [r"$\Omega = 0.07 - 0.3$",
                   r"DD p-value < 0.05",
                   r"Checkmate r > 1"], fontsize=15, loc=3)
        
    if i == 0:
        pass

        ax.set_ylabel(r"$\Delta(\chi_1^0, \chi_2^0)$ [GeV]", fontsize=12)

        sign = "\nM1<0"
        plt.text(np.max(x)*0.75, ax.get_ylim()[1]*0.4, 
                 "tanB: %i\nm_sl: %i GeV"% (tanB, m_sleptons) + sign*int(sign_M1 < 0), 
                 fontsize=14, bbox=dict(facecolor='white', edgecolor='black',))
        
        #blueline = ax.plot([],[], color="blue", alpha=0.4)[0]
        #ax.legend([blueline], ["M1 = M2"], loc=2, fontsize=15)
        
    if i == 5:
        plt.title(r"BR($\chi_1^{\pm} \to x_1^0 + \overline{q} + q$)", fontsize=16)
    

    if i == 6:
        plt.title("Checkmate r-value")
        accepted = cm_r0 < 1.
        cm_sc = plt.scatter(cm_x0[accepted], cm_y0[accepted], marker="o", color="green")
        cm_sc = plt.scatter(cm_x0[~accepted], cm_y0[~accepted], marker="o", color="red")
        scan_sc =  plt.scatter(x, y, marker="x", alpha=0.8)
        plt.legend([cm_sc, scan_sc], ["Checkmate Points", "Initial Scan points"])


#plt.plot(*get_allowed_polygon(x, y, om, dd).T)


i += 1
order = np.argsort(scan_data["M2"])
ax = f.add_subplot(2, 4, i+1)
ax.plot(np.array(scan_data["M2"])[order], np.array(scan_data["mu"])[order])
ax.set_xscale("log")
ax.set_title("M2 vs. mu", fontsize=15)
ax.set_xlabel("M2 [GeV]", fontsize=12)
ax.set_ylabel("mu [GeV]", fontsize=12)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, which="both")

    
plt.tight_layout()
plt.savefig("%s/CheckmateM2Scan_tanB_%i_mSl_%i.png" % (direc, tanB, m_sleptons))
plt.savefig("plots/CheckmateM2Scan_tanB_%i_mSl_%i.png" % (tanB, m_sleptons))
#plt.colorbar(plt.contourf([0,1],[0,1],[[0,0],[1,1]], np.linspace(0,1,11), cmap = "Reds"))

# Save, open image, and then remove it
if show_plot:
    img = Image.open("plots/CheckmateM2Scan_tanB_%i_mSl_%i.png" % (tanB, m_sleptons))
    img.show() 
