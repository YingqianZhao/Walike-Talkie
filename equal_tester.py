#Uses transform.py and padding_prob.py to test padding defense

CLOSED_SITENUM = 100
CLOSED_INSTNUM = 90
OPEN_SITENUM = 7000
INPUT_LOC = "datasantrim/"
OUTPUT_LOC = "equal_tester.results"

from equal_padding_prob import *
from transform import *

import time as systime
import subprocess, sys

def overhead(d, mpairs_data):
    out_size = 0
    in_size = 0
    out_padsize = 0
    in_padsize = 0
    for s in mpairs_data:
        for seq in s:
            for b in seq:
                out_size += b[0]
                in_size += b[1]
            padseq = d.pad(seq)

            for b in padseq:
                out_padsize += b[0]
                in_padsize += b[1]

    return (out_padsize + in_padsize)/float(out_size + in_size)

try: 
    OUTPUT_LOC = sys.argv[1]
    dType = sys.argv[2]
except:
    print "warning: using default OUTPUT_LOC", OUTPUT_LOC
    dType = "nofake"
    
#generate the defense
d = equal_defense()
d.load_example()

#Data format used is mpairs_data
#mbursts_data[i] is one site
#mbursts_data[i][j] is one sinste

#load the data
##print "Loading data..."
mpairs_data = []
for i in range(0, CLOSED_SITENUM):
    mpairs_data.append([])
    for j in range(0, CLOSED_INSTNUM):
        fname = str(i) + "-" + str(j) + ".burst"
        mbursts = read_mbursts(INPUT_LOC + fname)
        mpairs_data[-1].append(mbursts_to_mpairs(mbursts))

mpairs_data.append([]) #later put open world here
for i in range(0, OPEN_SITENUM):
    fname = str(i) + ".burst"
    mbursts = read_mbursts(INPUT_LOC + fname)
    mpairs_data[-1].append(mbursts_to_mpairs(mbursts))

logname = OUTPUT_LOC + ".log"
fout = open(logname, "w")
fout.close()
    
for opt_time in range(0, 400):
    print systime.time()

    add_cost = random.uniform(0.1, 2)
    fake_cost = random.uniform(0.1, 2)

    d.optimize_add(add_cost, mpairs_data)

    if dType == "nofake":
        d.fake_prob = 0
    else:
        if dType == "equal":
            force_equal = 1
            sdType = "uniform"
        else:
            force_equal = 0
            sdType = dType
        d.optimize_fake(fake_cost, mpairs_data, sdType, force_equal)
        d.fake_prob = random.uniform(0.1, 0.4)
        d.fake_max = random.randint(2, 10)
        
    true_chances = []
    ftrue_chances = []
    #evaluate a few trials
    for trial in range(0, 100):
        #we pick pad sequences randomly
        sequence = []
        while (sequence == []):
            pad_i = random.randint(0, CLOSED_SITENUM-1)
            pad_j = random.randint(0, CLOSED_INSTNUM-1)
            sequence = mpairs_data[pad_i][pad_j]

        padsequence = d.pad(sequence)

        logp = d.logprob_pad(padsequence, sequence)
        if (logp == 1):
            print "error: padsequence impossible"
            sys.exit(0)

        #true class is:
        true_class = pad_i

        #class_probs will hold the probability estimate of each class
        class_probs = []
        for i in range(0, CLOSED_SITENUM):
            class_probs.append(0)

        #closed world:
        for i in range(0, CLOSED_SITENUM):
            for j in range(0, CLOSED_INSTNUM):
                if (mpairs_data[i][j] != []):
                    logprob = d.logprob_pad(padsequence, mpairs_data[i][j])
                    if logprob != 1:
                        class_probs[i] += math.pow(math.e, logprob)

        #open world
        class_probs.append(0)
        for i in range(0, OPEN_SITENUM):
            seq = mpairs_data[-1][i]
            if (seq != []):
                logprob = d.logprob_pad(padsequence, seq)
                if logprob != 1:
                    class_probs[-1] += math.pow(math.e, logprob)



        if class_probs[true_class] == max(class_probs):
            true_chances.append(1.0)
        else:
            true_chances.append(0.0)

        ftrue_chance = class_probs[true_class] / sum(class_probs)
        ftrue_chances.append(ftrue_chance)
        
    oname = OUTPUT_LOC + "_" + str(opt_time)
    d.write_file(oname)

    import numpy
    fout = open(logname, "a")
    fout.write(oname + "\n")
    fout.write("Overhead: " + str(overhead(d, mpairs_data)) + "\n")
    fout.write("Mean accuracy: " + str(numpy.mean(true_chances)) + "\n")
    fout.write("False accuracy: " + str(numpy.mean(ftrue_chances)) + "\n")
    fout.write("\n")
    fout.close()
