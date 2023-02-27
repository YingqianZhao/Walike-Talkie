#Uses transform.py and padding_prob.py to test padding defense

CLOSED_SITENUM = 100
CLOSED_INSTNUM = 90
OPEN_SITENUM = 7000
INPUT_LOC = "datasantrim/"
OUTPUT_LOC = "equal_tester-det.results"
perburst = 1 #whether or not to use a different rounding set for each burst
#never tested perburst = 0

from padding_prob import *
from transform import *
from cclass import *
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

def seq_size(seq):
    #return total size of seq
    size = 0
    for s in seq:
        size += s[0]
        size += s[1]
    return size

def seq_dist(seq1, seq2):
    #len(seq1) >= len(seq2)
    if len(seq2) > len(seq1):
        seq1, seq2 = seq2, seq1


    dist = 0
    for i in range(0, len(seq2)):
        for b in range(0, 2):
            m = max(seq1[i][b], seq2[i][b])
            dist += m - seq1[i][b]
            dist += m - seq2[i][b]

    for i in range(len(seq2), len(seq1)):
        for b in range(0, 2):
            dist += seq1[i][b]
            dist += seq1[i][b]

    return dist

def seq_merge(seq1, seq2):
    merge_seq = []

    for i in range(0, max(len(seq1), len(seq2))):
        if i < len(seq1):
            burst1 = seq1[i]
        else:
            burst1 = [0, 0]
        if i < len(seq2):
            burst2 = seq2[i]
        else:
            burst2 = [0, 0]

        merge_seq.append([max(burst1[0], burst2[0]), max(burst1[1], burst2[1])])

    return merge_seq

try: 
    OUTPUT_LOC = sys.argv[1]
    perburst = int(sys.argv[2])
    if not(perburst in [0, 1]):
        raise NameError("perburst " + str(perburst))
except:
    print "warning: equal_tester-det.py using default OUTPUT_LOC", OUTPUT_LOC

#generate the defense
d = detdefense()
d.load_example()

#load the data:
#Data format used is mpairs_data
#mpairs_data[i] is one site
#mpairs_data[i][j] is one sinste
##print "Loading data..."
mpairs_data = []

#closed world
for i in range(0, CLOSED_SITENUM):
    mpairs_data.append([])
    for j in range(0, CLOSED_INSTNUM):
        fname = str(i) + "-" + str(j) + ".burst"
        mbursts = read_mbursts(INPUT_LOC + fname)
        mpairs_data[-1].append(mbursts_to_mpairs(mbursts))

#open world
mpairs_data.append([])
for i in range(0, OPEN_SITENUM):
    fname = str(i) + ".burst"
    mbursts = read_mbursts(INPUT_LOC + fname)
    mpairs_data[-1].append(mbursts_to_mpairs(mbursts))

logname = OUTPUT_LOC + ".log"
fout = open(logname, "w")
fout.close()

total_size = 0
for site in mpairs_data:
    for seq in site:
        total_size += seq_size(seq)
    
for opt_time in range(0, 10000):
    print systime.time()
    
    if perburst == 0:
        stepnum = 2
        d.optimize_det(0, stepnum, mpairs_data)

        stepnum = 2
        d.optimize_det(1, stepnum, mpairs_data)
    else:
        rmin = random.randint(2, 5)
        rmax = rmin + random.randint(0, 2)
        stepnums = [0] * 100
        for i in range(0, len(stepnums)):
            stepnums[i] = random.randint(rmin, rmax)
        d.optimize_det_perburst(0, stepnums, mpairs_data) #outgoing
        
        stepnums = [0] * 100
        for i in range(0, len(stepnums)):
            stepnums[i] = random.randint(rmin, rmax)
        d.optimize_det_perburst(1, stepnums, mpairs_data)

    true_chances = []

    collision_sequences = []
    collision_classes = []

    #pad everything
    for i in range(0, CLOSED_SITENUM+1):
        for sequence in mpairs_data[i]:
            padsequence = d.pad(sequence)
            if padsequence in collision_sequences:
                ind = collision_sequences.index(padsequence)
                collision_classes[ind].append(i)
            else:
                collision_sequences.append(padsequence)
                collision_classes.append([i])

    acc = cclass_to_acc(collision_classes)

    print "acc", acc

    pad_total_size = 0
    for seq_i in range(0, len(collision_sequences)):
        seq = collision_sequences[seq_i]
        pad_total_size += seq_size(seq) * len(collision_classes[seq_i])
    

    #now calculate true chance of each class
    #change this when attacker behavior changes for open world

    import numpy
    fout = open(logname, "a")
    fout.write("Overhead: " + str(pad_total_size/float(total_size)) + "\n")
    fout.write("Mean accuracy: " + str(acc) + "\n")
    fout.write("\n")
    fout.close()
