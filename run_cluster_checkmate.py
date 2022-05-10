from susy_tools import *
import numpy as np
import json
import argparse

use_pythia = True
remove_checkmate = True

# Parse filename arg
parser = argparse.ArgumentParser()
parser.add_argument("index", metavar="i", type=int)
args = vars(parser.parse_args())
i = args["index"]

with open("cluster_index_reference.txt", "r") as f:
    reference = json.load(f)
    
    direc = reference["directory"][i]
    slhafile = reference["filename"][i]
    
outfile = "checkmate%s" % (slhafile[slhafile.find("_"):])
if outfile not in os.listdir("%s/checkmate" % direc):

    if "checkmate_logs" not in os.listdir("."):
        os.mkdir("checkmate_logs")

    if "checkmate_%i.log" % i in os.listdir("checkmate_logs"):
        os.system("rm checkmate_logs/checkmate_%i.log" % i)

    cwd = os.getcwd()

    os.system("cp %s/spectra_slha/%s %s/bin/spectrum_%i.dat" % (direc, slhafile, checkmate_dir, i))
    os.chdir("%s/bin" % checkmate_dir)
    
    if use_pythia:
    	os.system("python runPythiaCheckmate.py spectrum_%i.dat >> %s/checkmate_logs/checkmate_%i.log" % (i, cwd, i))
    else:
        os.system("python runCheckmate.py spectrum_%i.dat >> %s/checkmate_logs/checkmate_%i.log" % (i, cwd, i))
    fileinfo = slhafile.split(".")[0].split("_")
    os.system("cp ../results/spectrum_%i/result.txt %s/%s/checkmate/%s" % (i, cwd, direc, outfile))
    os.system("cp -r ../results/spectrum_%i/evaluation %s/%s/checkmate/evaluation_%s" % (i, cwd, direc, outfile))

    if remove_checkmate:
        os.system("rm -rf ../result/spectrum_%i" % i)

    os.system("rm spectrum_%i.dat" % i)
    os.chdir(cwd)
