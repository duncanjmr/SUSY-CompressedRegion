from susy_tools import *
import numpy as np
import json
import argparse


# Parse filename arg
parser = argparse.ArgumentParser()
parser.add_argument("scan_dir", metavar="s", type=str)
args = vars(parser.parse_args())
direc = args["scan_dir"]

# Opening JSON file
with open("%s/checkmate_points.json" % direc) as json_file:
    data = json.load(json_file)
    pass
 
points = np.array(data["points"])

## Edit susy-hit input to generate spectra
changeParamValue("tanbeta(MZ)", data["tanB"])

changeParamValue("M_eL",   data["m_sleptons"])
changeParamValue("M_eR",   data["m_sleptons"])
changeParamValue("M_muL",  data["m_sleptons"])
changeParamValue("M_muR",  data["m_sleptons"])
changeParamValue("M_tauL", data["m_sleptons"])
changeParamValue("M_tauR", data["m_sleptons"])
    
def additional_command(M1, M2, index):
    changeParamValue("mu(EWSB)", data["mu"][index])
    
# Run susyhit and micromegas
cmdata = run(points, False, run_prospino=False, 
               run_micromegas=True,
               run_checkmate=True,
               working_directory="./%s/checkmate" % direc,
               additional_command=additional_command)

for k in ["tanB", "m_sleptons", "sign_M1", "mu"]:
    cmdata[k] = data[k]

cm_file =  "%s/checkmate_complete.json" % (direc)
with open(cm_file, "w") as outfile:
    json.dump(cmdata, outfile)
    
print()
