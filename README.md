# SUSY-CompressedRegion

--------------------------------------------------------------------
INSTALLATION

The current setup procedure is the following:

1. Set up susy-hit, micromegas, and checkmate, and copy their directory paths into the susy_tools.py file.

2. Copy get_gm2.C to the 'MSSM' folder in your micromegas installation. Then run 

	make main=get_gm2.C

This creates an executable file get_gm2 which takes a susyhit slha file as a command-line argument.

3. Copy default_checkmate.in and runCheckmate.py to the 'bin' folder in your checkmate installation. runCheckmate.py has one path that needs to be updated.

---------------------------------------------------------------------
WORKFLOW

generate_runpoints.py : 
	Edit this file with the parameters of interest, then run it with python3 to generate evenly spaced points in the compressed region. These can then be fed to checkmate for the best coverage and resolution of the area of interest. Output is a directory containing the information of the initial parameter space scan and the checkmate points.

prepare_for_cluster.py :
	This file takes arguments of the names of the outputs of the generate_runpoints.py, and creates a text file which is a reference for the checkate function which the cluster utilizes.

run_cluster_checkmate.py : 
	takes a single index as input, which it cross checks with the reference produced by 'prepare_for_cluster.py', and runs checkmate for the assigned point.


