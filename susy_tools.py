import numpy as np
import matplotlib.colors as colors
import os
import multiprocessing as mp
import queue
import time
import datetime
import warnings
from scipy.interpolate import interp2d
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


import subprocess as sp
from subprocess import DEVNULL, STDOUT, check_call


scripts_dir = "/home/duncan/UChicago/SUSY/compressed"
checkmate_dir = "/home/duncan/Software/checkmate2"
susyhit_dir = "/home/duncan/UChicago/SUSY/new_susyhit"
micromegas_dir = "/home/duncan/UChicago/SUSY/micromegas"

def changeParamValue(label, value):
    """
    Changes a parameter value in suspect2_lha.in.
    
    label is the comment attached to the line which is to be edited in the susyhit input file.
    """
    
    with open(susyhit_dir + "/suspect2_lha.in", "r") as f:
        s = f.read()

    end = s.find(' # ' + label)
    
    if end == -1:
        print("Finding parameter %s failed." % label)
        
    else:
        start = s[:end].rfind("\n") + 2

        line = s[start:end]

        if line[0] == "#": line[0] = " "

        parind = len(line) - len(line.lstrip())
        ind_val = len(line) - len(line[parind+3:].lstrip())
        line = line[:ind_val] + "{:.6e}".format(value)
        s = s[:start] + line + s[end:]
        with open(susyhit_dir + "/suspect2_lha.in", "w") as f:
            f.write(s)

            
def getParamValue(s, label):

    """
    Gets a parameter value, labeled by the comment (hashtag not included), in the susyhit_slha.out file.
    Note that one has to open the file and read it as a string before feeding it to this function.
    """
    end = s.find(' # ' + label)
    
    if end == -1:
        #warnings.warn("Finding parameter %s failed. Returning NaN." % label)
        return np.nan
        
    else:
        start = s[:end].rfind("\n") + 2

        line = s[start:end]

        if line[0] == "#": line[0] = " "

        
        return float(line.split()[-1])
    
def getBranchingRatio(s, label):
    """
    Similar to last function but instead of reading a parameter it reads a branching ratio.
    """
    
    end = s.find(' # ' + label)
    
    if end == -1:
        #warnings.warn("Finding parameter %s failed. Returning NaN." % label)
        return np.nan
        
    else:
        start = s[:end].rfind("\n") + 2

        line = s[start:end]

        if line[0] == "#": line[0] = " "

        parind = len(line) - len(line.lstrip())
        parlen = line[parind:].find(" ")
        
        ind_val = len(line) - len(line[parind+parlen:].lstrip())
        return float(line[parind: parind+parlen])
    

def generate_parameter_points(M1_max, M2_max, n_M1=8, n_M2=8, randomize=False, M1_min=10, M2_offset=0):
    """ 
    Generates a grid of parameter points in the (M1, M2) plane.
    
    These points are constrained by M2 > M1 + M2_offset, and M1 > M1_min
    """
    M1_sign = np.sign(M1_max)
    M2_sign = np.sign(M2_max)
    
    M1_max = np.abs(M1_max)
    M2_max = np.abs(M2_max)
    
    M1_l = np.linspace(M1_min, M1_max, n_M1)
    
    dM1 = (M1_max - M1_min) / n_M1
    dM2 = (M2_max - M2_offset) / n_M2

    first = True
    for M1 in M1_l:
        
        mn = M1 + M2_offset
        
        N = int(np.ceil(n_M2 * (M2_max - mn)/ (M2_max-M2_offset)))
        M2_l = np.abs(np.logspace(*np.log10([mn, M2_max]), N) + 
                      randomize * np.random.uniform(-dM2/2, dM2/2, N))
        
        p = np.vstack((M1_sign * np.abs(np.ones(N) * M1 + 
                                        randomize * np.random.uniform(-dM1/2, dM1/2, N)), 
                       M2_sign * M2_l)).T    
        if first:
            points = p
            first = False
        else:
            points = np.vstack((points, p))
            
    return points

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Given a color map, returns a new colormap which has been sliced between the two values. 
    One can choose a subsection of a colormap for more pretty plots.
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_higgs(A_t):
    """
    Runs susy-hit and reads the higgs mass, for a specific value of A_t, 
    the primary parameter for adjusting higgs mass.
    """
    changeParamValue("A_t", A_t)
    os.chdir("../new_susyhit")
    os.system("./run >> ../scripts/temp.out")
    
    # Get particle masses from susyhit output
    with open("susyhit_slha.out", "r") as f:
        s = f.read()
        h = getParamValue(s, "h")
    os.chdir("../scripts")
    return h

def optimize_higgs(h_target=125., tolerance=0.5):
    """
    Optimizes the higgs mass to a target value (usuall 125 GeV). 
    Useful if your parameters affect the higgs mass and you want to correct it.
    """
    
    x_0 = 3.0e3
    step = tolerance / h_target * x_0 / 2
    
    x = [x_0]
    h = [get_higgs(x_0)]

    c = 0
    while np.abs(h[-1] - h_target)  > tolerance and c < 5:
        hp = get_higgs(x[-1] + step)
        dh_dx = -(hp - h[-1]) / step
        
        if dh_dx == 0:
            break
        
        dx = (h[-1] - h_target) / dh_dx
        if abs(dx) > 1e3:
            dx = dx / abs(dx) * 1e3
        x.append(x[-1] + dx)
        h.append(get_higgs(x[-1]))
        c += 1

    return x[-1], h[-1]

def get_neutralino_eigenvalues(m1, m2, tanB, mu, ordered=True):

    sw = np.sqrt(0.222)
    cw = np.sqrt(1-sw**2)

    beta = np.arctan(tanB)
    MZ = 92.

    Massmatrix = np.array([[m1*cw**2 + m2 * sw**2, (m2-m1) *sw*cw, 0, 0],
                           [(m2-m1)*sw*cw, m1*sw**2 + m2*cw**2, MZ, 0],
                           [0, MZ, mu*np.sin(2*beta), -mu*np.cos(2*beta)],
                           [0,  0, -mu*np.cos(2*beta), -mu*np.sin(2*beta)]])


    masses = np.linalg.eig(Massmatrix)[0]
    
    if not ordered:
        return masses
    
    order = np.argsort(np.abs(masses))
    return masses[order]

def get_neutralinoMassDiff(M1, M2):
    """
    For given value of M1, M2, returns delta(x10, x20).
    """
    changeParamValue("M_1", M1)
    changeParamValue("M_2", M2)

    os.chdir(susyhit_dir)
    os.system("./run >> " + scripts_dir + "/temp.out")
    
    # Get particle masses from susyhit output
    with open("susyhit_slha.out", "r") as f:
        s = f.read()
        x2 = getParamValue(s, "~chi_20")
        x1 = getParamValue(s, "~chi_10")
        
    os.chdir(scripts_dir)
    return np.abs(x2) - np.abs(x1)

def get_neutralinoMasses(M1, M2):
    """
    For given value of M1, M2, returns delta(x10, x20).
    """
    changeParamValue("M_1", M1)
    changeParamValue("M_2", M2)

    os.chdir(susyhit_dir)
    os.system("./run >> " + scripts_dir + "/temp.out")
    
    # Get particle masses from susyhit output
    with open("susyhit_slha.out", "r") as f:
        s = f.read()
        x2 = getParamValue(s, "~chi_20")
        x1 = getParamValue(s, "~chi_10")
        
    os.chdir(scripts_dir)
    return np.array([np.abs(x1), np.abs(x2)])
        

def minimize_neutralinoMassDiff(M1_0, M2_0, to_minimize="M1", step = 0.5, return_diff=False, verbose=False):
    """
    Uses a linear approximation to find the M2 value which minimizes delta(x10, x20)
    """

    if to_minimize == "M1":
        ind = 0
        to_min = np.array([1,0])
    elif to_minimize == "M2":
        ind = 1
        to_min = np.array([0,1])
    else:
        raise Exception("to_minimize must be either \"M1\" or \"M2\"")

    step = 0.5
    M = [np.array([M1_0, M2_0])]
    diffs = [get_neutralinoMassDiff(*M[0])]

    dM = 0.1 * to_min
    M.append(M[0] + dM)
    diffs.append( get_neutralinoMassDiff(*M[-1]) )

    deriv = [10]

    dx = M[-1][ind] - M[-2][ind]
    deriv.append((diffs[-1] - diffs[-2])/dx)
    
    c = 0
    
    while np.abs(deriv[-1]) > 0.05 and np.abs(deriv[-2]) > 0.05  and diffs[-1] > 0.1 and c < 5:

        dM = -deriv[-1]*step*diffs[-1] * to_min
        M.append(M[-1] + dM)
        diffs.append(get_neutralinoMassDiff(*M[-1]))
        dx = M[-1][ind] - M[-2][ind]
        deriv.append((diffs[-1] - diffs[-2]) / dx)
        
        if verbose:
            print(M[-1], diffs[-1])
        
        c += 1
    
    est = M[-1]
    
    
    if verbose:
        print(M[-1], diffs[-1])
    
    if np.abs(deriv[-1]) < 0.8:
        
        M.append(M[-1] + dM)
        diffs.append(get_neutralinoMassDiff(*M[-1]))
        l3 = np.array(M).T[ind][-3:]

        #print(l3)
        #print(diffs[-3:])
        p, _ = curve_fit(lambda x, a, b, c: a * (x - b)**2 + c, l3, diffs[-3:], [0.05, l3[-1], diffs[-1]])
        #print(p)
        
        est[ind] = p[1]
    
    elif deriv[-1] > 0.8 and diffs[-1] < 1:
        est[ind] = est[ind] - diffs[-1] / deriv[-1]

        
    if return_diff:
        return est, get_neutralinoMassDiff(*est)
    else:
        return est

def get_approxGm2(M1, M2, mu, m_sl, tanB):
    alpha = 1./137
    sw_sq = 0.25

    m_mu = 0.1
    
    fxp = lambda x: (x**2 - 4*x + 3 + 2*np.log(x)) / (1-x)**3
    fx0 = lambda x: (x**2 - 1 - 2*x*np.log(x)) / (1-x)**3
    df_dx = lambda x : x/(1-x)**3 * (2*x - 2*(1 + np.log(x)) + 3 * (1-x)**2 * fx0(x))

    
    a_x0 = -alpha * M1 * m_mu**2 * mu * tanB / (4*np.pi*(1-sw_sq)*m_sl**4) * (fx0((M1/m_sl)**2) + df_dx((M1/m_sl)**2))
    a_xp = alpha*m_mu**2 * mu * M2 * tanB / (4*np.pi*sw_sq*m_sl**2) * (fxp((M2/m_sl)**2) - fxp((mu/m_sl)**2)) / (M2**2 - mu**2)
    
    return a_x0 + a_xp
    
def optimize_sleptonMassesGm2(M1, M2, mu, tanB, max_m_sl=2005., N=200, verbose=False):

    changeParamValue("mu(EWSB)", mu)
    changeParamValue("tanbeta(MZ)", tanB)
        
    def get_gm2(m_sleptons):
        
        changeParamValue("M_eL",   m_sleptons)
        changeParamValue("M_eR",   m_sleptons)
        changeParamValue("M_muL",  m_sleptons)
        changeParamValue("M_muR",  m_sleptons)
        changeParamValue("M_tauL", m_sleptons)
        changeParamValue("M_tauR", m_sleptons)
        
        queue = mp.Queue()
        run_once(M1, M2, True, queue, working_directory="test")
        return queue.get()["gm2"]
    
    xtest = np.logspace(np.log10(np.abs(M2)+10), np.log10(max_m_sl), N)
    gm2_est = get_approxGm2(M1, M2, mu, xtest, tanB)

    def get_best(xtest, gm2_est):
        
        #best = np.argmin(np.abs(gm2_est*1e9-2.74))
        score = (gm2_est*1e9-2.74)

        candidates = []
        for i in range(0, len(xtest)-1):
            if np.sign(score[i]) != np.sign(score[i+1]):
                candidates.append(np.interp(0, [score[i+1], score[i]], [xtest[i+1], xtest[i]]))

        if len(candidates) > 0:
            return np.max(candidates), 2.74e-9
        
        else:
            cand = np.argmin(np.abs(score))
            if np.abs(score[cand]) < 0.73:
                return xtest[cand], gm2_est[cand]
            else:
                return np.nan, np.nan
                
    
    
    initial, gm2_init = get_best(xtest, gm2_est)
    
    if np.isnan(initial): 
        print("Warning: Unable to satisfy g-2 with these parameters: (M1, M2, mu, tanB) = (%i, %i, %i, %i)" % (M1, M2, mu, tanB))
        
        ind = np.argmin(np.abs(gm2_est*1e9-2.74))
        print("\t Closest point: (m_sleptons, delta-g) = (%i, %.2E)" % (xtest[ind], gm2_est[ind]))
        return np.nan
    
    gm2_susyhit = get_gm2(initial)
    correction_size = gm2_susyhit / gm2_init
    
    #print(gm2_susyhit, correction_size)
    
    return get_best(xtest, correction_size * gm2_est)[0]
    


def run_once(M1, M2, remake, out_queue, run_prospino=False, 
             run_micromegas=True, run_checkmate=False, working_directory=".", 
             additional_command=None, index=None, masses_to_save=["~chi_10", "~chi_20"]):
    """
    Runs susyhit, micromegas, checkmate, and/or prospino once. Each program can be toggled.
    """
    
    # Set up
    filename = "spectrum_%i_%i.dat" % (M1, M2)
    
    cwd = os.getcwd()
    
    if isinstance(remake, bool):
        remake = {"prospino": remake, "susyhit": remake, "checkmate": remake}

    # Set param values to run over
    changeParamValue("M_1", M1)
    changeParamValue("M_2", M2)
    if additional_command != None:
        additional_command(M1, M2, index)
    
    # Run susyhit
    if filename not in os.listdir(working_directory + "/spectra_slha") or remake["susyhit"]:
        os.chdir(susyhit_dir)
        os.system("./run >> " + scripts_dir + "/temp.out")
        os.chdir(cwd)
        os.system("cp " + susyhit_dir + "/susyhit_slha.out " + working_directory + "/spectra_slha/" + filename)
        
    # Get particle masses from susyhit output
    with open(working_directory + "/spectra_slha/" + filename, "r") as f:
        s = f.read()
        if np.isnan(getParamValue(s, "EWSB")):
            out = {"index": index, "M1": M1, "M2": M2, "cx": np.nan, "r": np.nan, "analysis": np.nan }
            for var in ["gm2", "omega_dm", "dd_pval"] + masses_to_save:
                out[var] = np.nan
                
            out_queue.put(out)
            return None
              
    
        
    # Generate Prospino cross sections
    if run_prospino:
        outname = "cx_%i_%i.dat" % (M1, M2)
        if outname not in os.listdir(working_directory + "/prospino_cx") or remake["prospino"]:
            os.chdir("../prospino")
            os.system("cp " + scripts_dir + "/" + working_directory + "/spectra_slha/" + filename + 
                      " ./prospino.in.les_houches")
            try:
                os.system("./prospino_2.run >> " + scripts_dir + "/temp.out")
            except e:
                print("Error occurred when running prospino. Setting cx=nan.")
                out_queue.put({"m_x1": m_x1, "m_x2": m_x2, "cx": np.nan})
                raise e

            os.system("cp ./prospino.dat " + scripts_dir + "/" + working_directory + "/prospino_cx/" + outname)
            os.chdir(scripts_dir)

        with open(working_directory + "/prospino_cx/" + outname) as f:
            s = f.read()
            l = s.split("\n")[0].rstrip()
            cx = float(l[l.rfind(" "):])    
    else:
        cx = np.nan
                
    if run_checkmate:
        outname = "checkmate_%i_%i.dat" % (M1, M2)
        
        if outname not in os.listdir(working_directory + "/checkmate") or remake["checkmate"]:
            os.system("cp " + working_directory + 
                      "/spectra_slha/" + filename + 
                      " " + checkmate_dir + "/bin/")
            
            os.chdir(checkmate_dir + "/bin")
            os.system("python runCheckmate.py " + filename)
            
            os.system("rm " + filename)
            os.chdir(cwd)
           
            os.system("cp " + checkmate_dir + "/results/" + filename.split(".")[0] + "/result.txt " + working_directory + "/checkmate/" + outname)

            #os.system("rm -rf ../results/" + outname)
            
        # Read checkmate output and get r value
        with open(working_directory + "/checkmate/" + outname) as f:
            s=f.readlines()
            r = np.nan
            for l in s:
                line = l.split(": ")
                if line[0] == "Result for r":
                    r = float(line[1])
                if line[0] == "Analysis":
                    analysis = line[1]
            
                
              
    else:    
        r = np.nan
        analysis = ""
        
    outdir = {"index": index, "M1": M1, "M2": M2, "cx": cx, "r": r, "analysis": analysis}
    
    
    # Calculate g-2 contribution using micromegas
    if run_micromegas:
        s = os.popen(micromegas_dir + '/MSSM/get_gm2 ./' + working_directory + '/spectra_slha/' + filename).read()

        lines = s.split("\n")[:-1]

        for i, var in enumerate(["gm2", "omega_dm", "dd_pval"]):
            try:
                outdir[var] = float(lines[i].split(": ")[1])
            except:
                print("Couldn't get %s from: %s" % (var, lines[i]))
                outdir[var] = np.nan

        with open(working_directory + "/micromegas_out/micromegas_%i_%i.dat" % (M1, M2), "w") as f:
            f.write(s)
        
    # Get particle masses from susyhit output
    with open(working_directory + "/spectra_slha/" + filename, "r") as f:
        s = f.read()
        for label in masses_to_save:
            outdir[label] = getParamValue(s, label)
            
    out_queue.put(outdir)
    return None
    

def run(points_list, remake, run_prospino=False, run_micromegas=True, 
        run_checkmate=False,
        working_directory=None, verbose=True, additional_command=None, n_procs=3):
    
    """
    Over a list of [[M1, M2], [M1, M2], ...], runs susy programs and gives the output as a queue.
    Uses multiprocessing to speed up results.
    """
    
    if isinstance(remake, bool):
        remake = {"prospino": remake, "susyhit": remake, "checkmate": remake}
        
    if working_directory == None:
        working_directory = "output"
    
    # Set up directories
    Path(working_directory).mkdir(parents=True, exist_ok=True)
        
    for subdir in ["spectra_slha", "prospino_cx", "checkmate", "micromegas_out"]:
        if subdir not in os.listdir(working_directory):
            os.mkdir(working_directory + "/" + subdir)    
    
    processors = queue.Queue()

    out = mp.Queue()

    # Create processes, 3 at a time, to be run by cpus
    procs = []
    t0 = time.time()
    c = 1
    
    print("Running %i points... " % len(points_list))
    for i, p in enumerate(points_list):    
        if processors.qsize() > n_procs-1:
            proc = processors.get()
            proc.join()
            runtime = time.time() - t0
            remaining = (len(points_list) - c) * runtime / c
            if verbose:
                print("Process %i/%i Done! \t Runtime: "
                      "%s \t Time Remaining (est): %s \r" % (c, len(points_list), 
                             str(datetime.timedelta(seconds=int(runtime))),
                             str(datetime.timedelta(seconds=int(remaining)))), end="")
            c += 1
        
        
        proc = mp.Process(target=run_once, args=(p[0], p[1], remake, out, 
                                            run_prospino, run_micromegas,
                                            run_checkmate,
                                            working_directory, additional_command,
                                                 i))
        proc.start()
        processors.put(proc)
        time.sleep(0.1)

    while processors.qsize() > 0:
        proc = processors.get()
        proc.join()
        runtime = time.time() - t0
        remaining = (len(points_list) - c) * runtime / c
        
        if verbose:
            print("Process %i/%i Done! \t Runtime: "
                  "%s \t Time Remaining (est): %s \r" % (c, len(points_list), 
                         str(datetime.timedelta(seconds=int(runtime))),
                         str(datetime.timedelta(seconds=int(remaining)))), end="")
        c += 1
        
    data = {}
    ret = out.get()
    for k in ret.keys():
        data[k] = [ret[k]]
        
    while out.qsize() > 0:
        ret = out.get()
        for k in ret.keys():
            data[k].append(ret[k])
            
    
    #indsort = data.pop("index")
    #for k in data.keys():
    #    data[k] = [data[k][i] for i in indsort]
        
    return data
        
def draw_contour_old(run_points, uls, lumi=139., unit_conversion=1., cmap=None, 
                 alpha=1, show_ul_points=False, show_pointlabels=False):
    
    """
    Given a list of simulated cross section points and upper limit points, gives a sensitivity contour.
    
    Format of both run_points and uls is 
    [[m_x1_0, m_x2_0, cross_section_0],
     [m_x1_1, m_x2_1, cross_section_1],
     ...]
    """
    
    s = len(run_points) #-np.sqrt(2*sum(sel))
    tck_pts = interp.bisplrep(run_points[:,0], run_points[:,1], 
                          np.log(run_points[:,2]), s=s)

    tck_uls = interp.bisplrep(uls[:,0], uls[:,1], 
                          np.log(uls[:,2]), s=s)
    
    x = np.linspace(np.min(uls[:,0]), np.max(uls[:,0]), 20)
    y = np.linspace(np.min(uls[:,1]), np.max(uls[:,1]), 20)
    
    int_pts = interp.bisplev(x, y, tck_pts)
    int_uls = interp.bisplev(x, y, tck_uls) * np.sqrt(lumi / 139.) * unit_conversion

    if np.any(int_pts > int_uls):
        plt.contour(y, x, int_pts - int_uls, 
                       levels = [0, 10**20], cmap=cmap, 
                       alpha=alpha, linewidths=2)
        plt.contourf(y, x, int_pts - int_uls, 
                        levels = [-10**20, 0], cmap=cmap, alpha=alpha)
    col = cmap(0)

    """
    # Draw scatterplot with upper limit points, if requested
    if show_ul_points:
        plt.scatter(file[:,1], file[:,0], edgecolors=col, facecolor="none")

        sel2 = cx_interp > ul
        plt.scatter(file[sel2][:, 1], file[sel2][:, 0], color=col)

    if show_pointlabels:
        for i, l in enumerate(file):
            if l[1] < xmax and l[0] < ymax:
                plt.text(l[1]+xmax*0.01, l[0]+ymax*0.01, "%.2f" % ul[i], 
                         color=col, fontsize=11)
    """

def draw_contour(run_points, uls, lumi=139., unit_conversion=1., cmap=None, 
                 alpha=1, show_ul_points=False, show_pointlabels=False):
    
    """
    Given a list of simulated cross section points and upper limit points, gives a sensitivity contour.
    
    Format of both run_points and uls is 
    [[m_x1_0, m_x2_0, cross_section_0],
     [m_x1_1, m_x2_1, cross_section_1],
     ...]
    """
    
    s = len(run_points) #-np.sqrt(2*sum(sel))
    k = int(np.floor(np.sqrt(s)) - 1)
    if k > 3:
        k=3
    print(s,k)
    
    tck_pts = interp.bisplrep(run_points[:,0], run_points[:,1], 
                          np.log(run_points[:,2]))
    
    int_pts = []
    for l in uls:
        c = interp.bisplev(l[0], l[1], tck_pts)
        if c > 10: c=10
        int_pts.append( np.exp(c) )

    int_pts = np.array(int_pts)
    ul_adj =  uls[:,2] * np.sqrt(lumi / 139.) * unit_conversion
        
    if np.any(int_pts > ul_adj):
        plt.tricontour(uls[:,1], uls[:,0], int_pts - ul_adj, 
                       levels = [0, 10**20], cmap=cmap, 
                       alpha=alpha)
        plt.tricontourf(uls[:,1], uls[:,0], int_pts - ul_adj, 
                        levels = [0, 10**20], cmap=cmap, alpha=alpha)
    col = cmap(0)
    
    
    # Draw scatterplot with upper limit points, if requested
    if show_ul_points:
        plt.scatter(uls[:,1], uls[:,0], edgecolors=col, facecolor="none")

        sel = int_pts > ul_adj
        plt.scatter(uls[sel][:, 1], uls[sel][:,0], color=col)

    if show_pointlabels:
        for i, l in enumerate(uls):
            plt.text(l[1], l[0], "%.1f\n%.1f" % (ul_adj[i], int_pts[i]),
                                                     color=col, fontsize=8)
    