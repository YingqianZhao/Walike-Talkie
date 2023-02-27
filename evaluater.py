#Uses transform.py and padding_prob.py to test padding defense
#Input SHOULD be cleaned (i.e. don't take directly from experimentX)

CLOSED_SITENUM = 100
CLOSED_INSTNUM = 90
OPEN_SITENUM = 7000
INPUT_LOC = "datasantrim/"
OUTPUT_LOC = ""
DEFENSE_LOC = ""

from equal_padding_prob import *
from transform import *

import subprocess, sys, numpy, time

def overhead(d, mpairs_data):
    out_size = 0
    in_size = 0
    out_padsize = 0
    in_padsize = 0
    for trial in range(0, 10):
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
    DEFENSE_LOC = sys.argv[2]
except:
    print "call: python evaluater.py OUTPUT_LOC DEFENSE_LOC"
    sys.exit(0)
#load the defense
d = equal_defense()
d.read_file(DEFENSE_LOC)

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
fout.write(DEFENSE_LOC + "\n")
oh = overhead(d, mpairs_data)
fout.write("Overhead: " + str(oh) + "\n")
fout.close()

tpr_chances = []
tnr_chances = []

#evaluate many, many trials
for multi_trial in range(0, 100): #just to cut down on log overhead
    #tpr should not be reset here
    print time.time()
    for trial in range(0, 100):        
        #we pick pad sequences randomly
        is_open = random.randint(0, 1)
        
        sequence = []
        while (sequence == []):
            if (is_open == 0):
                pad_i = random.randint(0, CLOSED_SITENUM-1)
                pad_j = random.randint(0, CLOSED_INSTNUM-1)
            if (is_open == 1):
                pad_i = CLOSED_SITENUM
                pad_j = random.randint(0, OPEN_SITENUM-1)
            sequence = mpairs_data[pad_i][pad_j]            

        padsequence = d.pad(sequence)

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

                    
        guessed_class = class_probs.index(max(class_probs))

        if guessed_class < CLOSED_SITENUM:
            if guessed_class != true_class:
                tnr_chances.append(0.0)
            else:
                tpr_chances.append(1.0)
        if guessed_class == CLOSED_SITENUM:
            if guessed_class != true_class:
                tpr_chances.append(0.0)
            else:
                tnr_chances.append(1.0)
        
    fout = open(logname, "a")
    fout.write("Time: " + str(time.time()) + "\n")
    fout.write("Trialnum: " + str((multi_trial+1)*100) + "\n")
    fout.write("TPR: " + str(numpy.mean(tpr_chances)) + "\n")
    fout.write("TNR: " + str(numpy.mean(tnr_chances)) + "\n")
    fout.write("\n")
    fout.close()
