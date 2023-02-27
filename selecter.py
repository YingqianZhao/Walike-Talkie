#reads in all *.log files in some folder
#chooses 100 of the best given solutions
#outputs them to another folder in order

INPUT_LOC = "experiment10/"
OUTPUT_LOC = "testedefensesf/"

##defnames = ["tester_normal_perburst",
##            "tester_normal_allburst",
##            "tester_uniform_perburst",
##            "tester_uniform_allburst"]

defnames = ["etester_equal",
            "etester_nofake",
            "etester_normal",
            "etester_uniform"]

import random
import subprocess

for defname in defnames:

    cmd = "ls " + INPUT_LOC + defname + "*.log"
    s = subprocess.check_output(cmd, shell=True)
    s = s.split("\n")[:-1]
    
    files = []
    wanted_files = []
    #read in all data from log files
    for defname_i in s:
        print defname_i
        f = open(defname_i, "r")
        logdata = f.read()
        f.close()
        
        logdata = logdata.split("\n\n")[:-1]
        
        for logdata_onefile in logdata:
            logdata_onefile = logdata_onefile.split("\n")[:-1]
            filename = logdata_onefile[0]
            oh = float(logdata_onefile[1].split(":")[1])
            acc = float(logdata_onefile[2].split(":")[1])
            files.append([filename, oh, acc])

    #pareto it
    pareto_files = []
    files_ok = [1] * len(files)
    
    for f_i in range(0, len(files)):
        for f_j in range(0, len(files)):
            if (files[f_i][1] > files[f_j][1] and #oh larger
                files[f_i][2] > files[f_j][2]): #acc larger
                files_ok[f_i] = 0
                break

    for f_i in range(0, len(files)):
        if files_ok[f_i] == 1:
            pareto_files.append(files[f_i])

    files = sorted(files, key = lambda files:files[2])
    files = sorted(files, key = lambda files:files[1])

    #get 100 files, first from pareto
    if len(pareto_files) > 100:
        pareto_files = random.sample(pareto_files, 100)
    for f in pareto_files:
        wanted_files.append(f)

    #then from the rest
    still_wanted_num = 100 - len(wanted_files)
    if (still_wanted_num > 0):
        still_wanted_files = random.sample(files, still_wanted_num)
    for f in still_wanted_files:
        wanted_files.append(f)

    #sort again for lulz
    wanted_files = sorted(wanted_files, key = lambda files:files[2])
    wanted_files = sorted(wanted_files, key = lambda files:files[1])
        
    #now output
    count = 0
    for count in range(0, 100):
        infname = INPUT_LOC + wanted_files[count][0]
        outfname = OUTPUT_LOC + defname + "_" + str(count)
        cmd = "cp " + infname + " " + outfname
        subprocess.call(cmd, shell=True)
