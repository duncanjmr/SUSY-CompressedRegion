import multiprocessing as mp
import os
import json
import datetime
import time

filename = "cluster_index_reference.txt"
verbose = True

if filename not in os.listdir("./"):
    raise NameError("List of checkmate points not found. Run prepare_for_cluster.py")

with open(filename, "r") as f:
    reference = json.load(f)
    N = len(reference["filename"])

def run_checkmate(i):
    os.system("python3 run_cluster_checkmate.py %i" % i)


processors = []

print("Running %i points... " % N)
n_procs = 6
c = 1
t0 = time.time()
for i in range(N): 
    while len(processors) > n_procs-1:
        for j, proc in enumerate(processors):
            if not proc.is_alive():
                runtime = time.time() - t0
                if verbose:
                    print("Process %i/%i Done! \t Runtime: %s\r" % (c, N, 
                                 str(datetime.timedelta(seconds=int(runtime)))), end="")
                processors.pop(j)
                c += 1

        time.sleep(0.2)

    proc = mp.Process(target=run_checkmate, args=(i,))
    proc.start()
    processors.append(proc)
    time.sleep(0.1)

while len(processors) > 0:
    for j, proc in enumerate(processors):
        if not proc.is_alive():
            runtime = time.time() - t0
            if verbose:
                print("Process %i/%i Done! \t Runtime: %s\r" % (c, N, 
                             str(datetime.timedelta(seconds=int(runtime)))), end="")
            processors.pop(j)
            c+= 1
        time.sleep(0.2)

print()
print("Checkmate Runs Finished")

