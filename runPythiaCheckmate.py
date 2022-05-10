import argparse
import sys
import os

sys.path.append('/home/duncan/Software/madgraph3.2')
#import models.check_param_card as mgconverter
from models.check_param_card import convert_to_mg5card

# Parse filename arg
parser = argparse.ArgumentParser()
parser.add_argument("slha_file", metavar="f", type=str)

args = vars(parser.parse_args())

input_file = args["slha_file"]

# Convert file to MG acceptable
fin_read = open(input_file,'r').readlines()
fo = open(input_file + ".temp", 'w')
for line in fin_read:
    if '# PDG code           mass       particle' in line:
        fo.write(line)
        fo.write('       6 1.750000e+02 # MT \n')
        fo.write('      15 1.777000e+00 # Mta \n')
        fo.write('      23 9.118760e+01 # MZ \n')
 #elif 'DECAY 6' in line:
 #   fo.write(line)
 #   fo.write('DECAY  23 2.411433e+00 # WZ \n')
 #   fo.write('DECAY  24 2.002822e+00 # WW \n')
    elif '# br(t ->  b    w+)' in line:
        continue
    else:
        fo.write(line)
fo.close()

convert_to_mg5card(input_file + ".temp", "MG_" + input_file)
os.system("rm " + input_file + ".temp")


# Update a checkmate run file

cm_in = open("pythia_checkmate.in", "r")

cm_filename = "checkmate_" + input_file.split(".")[0] + ".in"
cm_in_updated = open(cm_filename, "w")

pythia_cards = []
for fname in ["pythia_card_pos.in", "pythia_card_sleptons.in", "pythia_card_neg.in"]:
    default = open(fname, "r")
    pythia_cards.append(fname.split(".")[0] + "_" + input_file.split(".")[0] + ".in")
    output = open(pythia_cards[-1], "w")
    for line in default:
        if line[:9] == "SLHA:file":
            output.write("SLHA:file = MG_" + input_file + "\n")
        else:
            output.write(line)
    output.close()
        
c = 0
        
for line in cm_in:
    if line[:8] == "SLHAFile":
        cm_in_updated.write("SLHAFile: /home/duncan/Software/checkmate2/bin/MG_" + input_file + "\n")
    elif line[:4] == "Name":
        cm_in_updated.write("Name: " + input_file.split(".")[0] + " \n")
    elif line[:11] == "Pythia8Card:":
        cm_in_updated.write("Pythia8Card: " + pythia_cards[i])
        c+=1
    else:
        cm_in_updated.write( line )

cm_in_updated.close()

orig = os.environ["PYTHONPATH"]
os.environ["PYTHONPATH"] = "/home/duncan/Software/python2/lib/python2.7/site-packages:/home/duncan/Software/root6/lib"
    
os.system("./CheckMATE " + cm_filename)
os.environ["PYTHONPATH"] = orig

os.system("rm " + "MG_" + input_file)
os.system("rm " + cm_filename)
for p in pythia_cards:
    os.system("rm " + p)
