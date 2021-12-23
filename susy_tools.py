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

import subprocess as sp
from subprocess import DEVNULL, STDOUT, check_call


scripts_dir = "/home/duncan/UChicago/SUSY/scripts"
checkmate_dir = "/home/duncan/Software/checkmate2"
susyhit_dir = "/home/duncan/UChicago/SUSY/new_susyhit"

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

        parind = len(line) - len(line.lstrip())
        parlen = line[parind:].find(" ")
        
        ind_val = len(line) - len(line[parind+parlen:].lstrip())
        return float(line[ind_val:])
    
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
    return x2 - x1


def get_closestmass(M1, return_diff=False, fast=False):
    """
    Uses a linear approximation to find the M2 value which minimizes delta(x10, x20)
    """
    dx = M1 * 0.2
    est_lo = M1 + dx - get_neutralinoMassDiff(M1, M1 + dx)
    est_hi = M1 + get_neutralinoMassDiff(M1, M1 - dx) - dx
    
    est = 0.5 * (est_lo + est_hi)
    
    if return_diff:
        return est, get_neutralinoMassDiff(M1, est)
    else:
        return est


def run_once(M1, M2, remake, out_queue, run_prospino=False, 
        run_micromegas=False, run_checkmate=False, working_directory=".", additional_command=None, index=None):
    """
    Runs susyhit, micromegas, checkmate, and/or prospino once. Each program can be toggled.
    """
    
    # Set up
    filename = "spectrum_%i_%i.dat" % (M1, M2)
    
    cwd = os.getcwd()
    
    if isinstance(remake, bool):
        remake = {"prospino": remake, "susyhit": remake, "checkmate": remake}

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
        m_x1 = getParamValue(s, "~chi_10")
        m_x2 = getParamValue(s, "~chi_20")
        h = getParamValue(s, "h")
        
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
                      " " + checkmate_dir + "/bin/susyhit_slha.out")
            
            os.chdir(checkmate_dir + "/bin")
            os.system("python3 converter.py")
            
            orig = os.environ["PYTHONPATH"]
            os.environ["PYTHONPATH"] = "/home/duncan/Software/python2/lib/python2.7/site-packages:/home/duncan/Software/root6/lib"
            
            maxev = 10000

            sp.run(['./CheckMATE', '--name', outname, '--analysis', 'Atlas13tev', 
                    '--slha-file', './susyhit_MGcard.dat', '-maxev', str(maxev),
                    '-mgcommand', 'import model MSSM_SLHA2; generate p p > n2 x1+; generate p p > n2 x1-', 
                    '-sp', '-oe', 'overwrite'], stdout=DEVNULL, stderr=STDOUT)

            os.environ["PYTHONPATH"] = orig
            os.chdir(cwd)
           
            os.system("cp " + checkmate_dir + "/results/" + outname + "/result.txt " + working_directory + "/checkmate/" + outname)

            
            #os.system("rm -rf ../results/" + outname)
            

            
        with open(working_directory + "/checkmate/" + outname) as f:
            s=f.readlines()
            r = np.nan
            for l in s:
                line = l.split(":")
                if line[0] == "Result for r":
                    r = float(line[1])
              
    else:    
        r = np.nan

    
    # Calculate g-2 contribution using micromegas
    s = os.popen('../micromegas/MSSM/get_gm2 ./' + working_directory + '/spectra_slha/' + filename).read()
    try:
        gm2 = float(s.split("\n")[0])
        om = float(s.split("\n")[1])
    except:
        gm2 = np.nan
        om = np.nan

    out_queue.put({"M1": M1, "M2": M2, "m_x1": m_x1, "m_x2": m_x2, "cx": cx, "gm2": gm2, "omega": om, "r": r})
    

def run(points_list, remake, run_prospino=False, run_micromegas=False, 
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
    if working_directory not in os.listdir("."):
        os.mkdir(working_directory)

    if "spectra_slha" not in os.listdir(working_directory):
        os.mkdir(working_directory + "/spectra_slha")    

    if "prospino_cx" not in os.listdir(working_directory):
        os.mkdir(working_directory + "/prospino_cx")   
        
    if "checkmate" not in os.listdir(working_directory):
        os.mkdir(working_directory + "/checkmate")   
    
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
        
        
    return out
        
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
    