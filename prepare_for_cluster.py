import argparse
import json
import os


# Parse filename arg
parser = argparse.ArgumentParser()
parser.add_argument("scan_dir", metavar="s", nargs="+", type=str)
args = vars(parser.parse_args())
direc = args["scan_dir"]
    
filename = "cluster_index_reference.txt"

i=0
reference = {"index": [], "directory": [], "filename": []}
for d in direc:
    for s in os.listdir("%s/checkmate/spectra_slha" % d):
        reference["index"].append(i)
        reference["directory"].append("%s/checkmate" % d)
        reference["filename"].append(s)
        i += 1
        
with open(filename, "w") as f:
    json.dump(reference, f)

