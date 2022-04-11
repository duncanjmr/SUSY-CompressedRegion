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
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import nearest_points


import subprocess as sp
from subprocess import DEVNULL, STDOUT, check_call


scripts_dir = "/home/duncan/UChicago/SUSY-CompressedRegion"
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
            print(M[-1], diffs[-1], deriv[-1])
        
        c += 1
    
    est = M[-1]
    

    M.append(M[-1] + dM)
    diffs.append(get_neutralinoMassDiff(*M[-1]))
    
    deriv2 = 4*(diffs[-1] - 2*diffs[-2] + diffs[-3]) / (M[-1][ind] - M[-3][ind])**2
    
    if verbose: print(deriv2)
    
    if deriv2 > 0.1 and deriv[-1] < 0.8:
        
        M.append(M[-1] + dM)
        diffs.append(get_neutralinoMassDiff(*M[-1]))
        
        l3 = np.array(M).T[ind][-4:]
        if verbose:
            print(l3, diffs[-4:])
        
        #print(l3)
        #print(diffs[-3:])
        p, _ = curve_fit(lambda x, a, b, c: a * (x - b)**2 + c, l3, diffs[-4:], [0.2, l3[np.argmin(diffs[-4:])], np.min(diffs[-4:])])        #print(p)
        
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
    
def optimize_gm2(M1, M2, mu, tanB, m_sleptons, max_test=3005., N=300, to_minimize="sleptons", verbose=False):

    changeParamValue("mu(EWSB)", mu)
    changeParamValue("tanbeta(MZ)", tanB)
        
    def get_gm2(M1, M2, mu, m_sleptons):
        
        changeParamValue("mu(EWSB)", mu)
        
        changeParamValue("M_eL",   m_sleptons)
        changeParamValue("M_eR",   m_sleptons)
        changeParamValue("M_muL",  m_sleptons)
        changeParamValue("M_muR",  m_sleptons)
        changeParamValue("M_tauL", m_sleptons)
        changeParamValue("M_tauR", m_sleptons)
        
        queue = mp.Queue()
        run_once(M1, M2, True, queue, working_directory="test")
        return queue.get()["gm2"]  
    
    if to_minimize == "sleptons":
        xtest = np.logspace(np.log10(np.abs(M2)+10), np.log10(max_test), N)
        gm2_est = get_approxGm2(M1, M2, mu, xtest, tanB)
        gm2_susyhit = lambda x: get_gm2(M1, M2, mu, x)
        
    elif to_minimize == "mu":
        xtest = np.logspace(np.log10(100), np.log10(max_test), N)
        gm2_est = get_approxGm2(M1, M2, xtest, m_sleptons, tanB)
        gm2_susyhit = lambda x: get_gm2(M1, M2, x, m_sleptons)

    def get_best(xtest, gm2_est):
        
        #best = np.argmin(np.abs(gm2_est*1e9-2.74))
        score = (gm2_est*1e9-2.74)

        candidates = []
        for i in range(0, len(xtest)-1):
            if np.sign(score[i]) != np.sign(score[i+1]) and np.abs(score[i]) < 0.5 and np.abs(score[i+1]) < 0.5:
                candidates.append(np.interp(0, [score[i+1], score[i]], [xtest[i+1], xtest[i]]))

        if len(candidates) > 0:
            if len(candidates) > 1:
                #print("Multiple gm2 candidates found: %s" % (str(candidates)))
                pass
            return np.min(candidates), 2.74e-9
        
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
    
    gm2_true = gm2_susyhit(initial)
    correction_size = gm2_true / gm2_init
    
    final, gm2_corr = get_best(xtest, correction_size * gm2_est)
    
    gm2_final = gm2_susyhit(final)
    
    if verbose:
        print("Tested %s range: \t%i - %i" % (to_minimize, min(xtest), max(xtest)))
        print("Tree level gm2 range: \t%.2E - %.2E" % (min(gm2_est), max(gm2_est)))
        print("Loop Correction Size: \t%.2f" % ((correction_size - 1) * 100) + "%")
        print("Initial %s, gm2 values: \t%i , %.2E" % (to_minimize, initial, gm2_true))
        print("Final %s, gm2 values: \t%i , %.2E" % (to_minimize, final, gm2_final))

    
    return final
        
    


def run_once(M1, M2, remake, out_queue, run_prospino=False, 
             run_micromegas=True, run_checkmate=False, working_directory=".", 
             additional_command=None, index=None, masses_to_save=["~chi_10", "~chi_20"]):
    """
    Runs susyhit, micromegas, checkmate, and/or prospino once. Each program can be toggled.
    """
    
    # Set up
    filename = "spectrum_%i_%i.dat" % (M1, M2)
    Path(working_directory).mkdir(parents=True, exist_ok=True)
    for subdir in ["spectra_slha", "prospino_cx", "checkmate", "micromegas_out"]:
        if subdir not in os.listdir(working_directory):
            os.mkdir(working_directory + "/" + subdir) 

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
        
        
    outdir = {"index": index, "M1": M1, "M2": M2}

    # Calculate g-2 contribution using micromegas
    dd_om_excluded = False
    if run_micromegas:
        s = os.popen(micromegas_dir + '/MSSM/get_gm2 ./' + working_directory + '/spectra_slha/' + filename).read()

        lines = s.split("\n")[:-1]

        for i, var in enumerate(["gm2", "omega_dm", "dd_pval"]):
            try:
                outdir[var] = float(lines[i].split(": ")[1])
            except:
                print("Couldn't get %s from: %s" % (var, lines))
                outdir[var] = np.nan
           
        with open(working_directory + "/micromegas_out/micromegas_%i_%i.dat" % (M1, M2), "w") as f:
            f.write(s)
                            
    if run_checkmate and not dd_om_excluded:
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
            os.system("cp -r " + checkmate_dir + "/results/" + filename.split(".")[0] + "/evaluation " + working_directory + "/checkmate/" + "evaluation_%i_%i" % (M1, M2))

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
        
        
    outdir["cx"] = cx
    outdir["r"] = r 
    outdir["analysis"] = analysis
    
        
    # Get particle masses from susyhit output
    with open(working_directory + "/spectra_slha/" + filename, "r") as f:
        s = f.read()
        for label in masses_to_save:
            outdir[label] = getParamValue(s, label)
            
    out_queue.put(outdir)
    return None
    

def run(points_list, remake, run_prospino=False, run_micromegas=True, 
        run_checkmate=False,
        working_directory=None, verbose=True, additional_command=None, n_procs=6):
    
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
    
    processors = []

    out = mp.Queue()

    # Create processes, 3 at a time, to be run by cpus
    t0 = time.time()
    c = 1
    
    print("Running %i points... " % len(points_list))
    n_completed = 0
    for i, p in enumerate(points_list): 
        while len(processors) > n_procs-1:
            for j, proc in enumerate(processors):
                if not proc.is_alive():
                    n_completed += 1
                    
                    runtime = time.time() - t0
                    if verbose:
                        print("Process %i/%i Done! \t Runtime: %s\r" % (c, len(points_list), 
                                     str(datetime.timedelta(seconds=int(runtime)))), end="")
                    processors.pop(j)
                    c += 1
            
            time.sleep(0.2)
        
        proc = mp.Process(target=run_once, args=(p[0], p[1], remake, out, 
                                            run_prospino, run_micromegas,
                                            run_checkmate,
                                            working_directory, additional_command,
                                                 i))
        proc.start()
        processors.append(proc)
        time.sleep(0.1)

    while len(processors) > 0:
        for j, proc in enumerate(processors):
            if not proc.is_alive():
                n_completed += 1

                runtime = time.time() - t0
                if verbose:
                    print("Process %i/%i Done! \t Runtime: %s\r" % (c, len(points_list), 
                                 str(datetime.timedelta(seconds=int(runtime)))), end="")
                processors.pop(j)
                c+= 1
            time.sleep(0.2)
        
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
        
    
############ Section of code for generating checkmate run points #################
import scipy.optimize as op

def costFunction(points, boundary, boundaryfactor = 1., widths=np.array([1.,1.]), totalcharge=1.):
    """ 
    A cost function which can be minimized with scipy to find the optimal point layout within 
    a set of boundary points.
    """
    n = len(points)
    
    distances = np.hstack([np.sqrt(np.sum(((points[j:] - points[:-j])/widths)**2, axis=1)) 
                           for j in range(1, len(points))])
    potential = totalcharge / n * np.sum(1./distances**2)
    return potential / n
    
def keep_insideboundary(boundary, r):
    
    if Polygon(boundary).contains(Point(r)):
        return r
    else:
        if np.any(boundary[0] != boundary[-1]):
            boundary = np.vstack((boundary, boundary[0]))

        dr = boundary[1:] - boundary[:-1]
        dr_mag = np.sqrt(np.sum(dr**2, axis=1))
        tval = np.sum((boundary[:-1] - r) * dr, axis = 1) / dr_mag**2

        tval[tval > 1.] = 1.
        tval[tval < 0] = 0

        dist = np.sqrt(np.sum(( dr * tval[:,None] + boundary[:-1] - r )**2, axis=1))
        idx = np.argmin(dist)
        return (dr * tval[:,None] + boundary[:-1])[idx]
    
def optimize_points(r0, boundary, cost_function, n_steps=50, return_full=False):
    
    dx_mag = 0.001
    stepsize = 0.02
    f = lambda r: costFunction(r, boundary)

    xp = [r0]

    cost = [f(r0)]
    minfound=False
    for step in range(n_steps):
        if minfound: 
            continue
            
        dc_dxi_0 = []
        for i in range(len(xp[-1])):
            dc_dxi_0.append([])
            for j in range(2):

                dx = np.zeros(xp[0].shape)
                dx[i, j] = dx_mag*stepsize
                dc_dxi_0[i].append( (f(xp[-1] + dx) - f(xp[-1]))/(dx_mag*stepsize))

        grad = np.array(dc_dxi_0)
        
        mags = np.sum(grad, axis=1)
        toobig = stepsize*mags > 0.05
        grad[toobig] = grad[toobig] / mags[toobig][:,None] * 0.05
        proposed = np.array([keep_insideboundary(boundary, r) for r in xp[-1] - stepsize * grad])
        
        co = f(proposed)
        
        if not minfound:
            xp.append(proposed)
            cost.append(f(xp[-1]))   
                
    if return_full:
        return xp
    else:
        return xp[-1]

    
def get_allowed_polygon(x, y, om, dd):
    """
    Generates the boundary points given the omega_dm and direct detection constraints.
    """
    com = plt.tricontourf(x,y,om, levels=[0.3, 1e10], alpha=0);
    om_bounds = [col.get_paths()[0].vertices.T for col in com.collections][0]
    
    cdm = plt.tricontourf(x,y,dd, levels=[-1,0.05], alpha=0);
    dd_bounds = [col.get_paths()[0].vertices.T for col in cdm.collections][0]

    mp = MultiPoint(np.array([x, y]).T)
    p_space = mp.buffer(1).buffer(-1)

    if len(dd_bounds.T) > 2:
        dd_pol = Polygon(dd_bounds.T)
        p_space = p_space.difference(dd_pol)
    if len(om_bounds.T) > 2:
        om_pol = Polygon(om_bounds.T)
        p_space = p_space.difference(om_pol)
    
    scan_region = p_space.buffer(-0.01).buffer(0.01)
    
    boundary_points = [scan_region.boundary.interpolate(a, normalized=True) for a in np.linspace(0, 1, 100)]
    boundary_coords = np.array([[b.x, b.y] for b in boundary_points])
    return boundary_coords

def generate_grid(number, boundary):
    """
    Generates a grid of N points within the boundary. Generally a good initial guess for minimization.
    """
    xmax, ymax = np.max(boundary, axis=0)*0.99
    xmin, ymin = np.min(boundary, axis=0)*1.01

    wx = xmax - xmin
    wy = ymax - ymin

    density = number / (wx * wy)
    area = Polygon(boundary).area

    Np = number * number/(area*density)
    N_axis = int(np.ceil(np.sqrt(Np)*1.2))

    n_inside = 10000
    
    while n_inside > number:
        g0 = np.meshgrid(np.linspace(xmin, xmax, N_axis), np.linspace(ymin, ymax, N_axis))
        p0 = np.vstack((g0[0].flatten(), g0[1].flatten()))

        n_inside = np.sum([Polygon(boundary).contains(Point(p)) for p in p0.T])
        xmax *= 1.005
        ymax *= 1.005

    p_out = []
    for p in p0.T:
        if Polygon(boundary).contains(Point(p)):
            p_out.append(p)

    return np.array(p_out)

def get_checkmatePoints(x, y, om, dd, N, totalcharge=1., return_full=False, n_steps=50):
    
    wx = np.max(x) - np.min(x)
    wy = np.max(y) - np.min(y)
    boundary = get_allowed_polygon(x/wx, y/wy, om, dd)

    plt.close()

    grid_points = generate_grid(N, boundary)
    widths = np.max(boundary, axis=0) - np.min(boundary, axis=0)
    scan_points = optimize_points(grid_points, boundary, costFunction, n_steps=n_steps, 
                                  return_full=return_full)

    return np.array(scan_points)*np.array([wx,wy]), boundary* np.array([wx, wy])


