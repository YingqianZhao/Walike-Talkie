import sys
import random
from transform import *
from cclass import *
import time as systime

#produces merge.data in the graphs folder

fold = "datasantrim/"
CLOSED_SITENUM = 100
CLOSED_INSTNUM = 90
OPEN_SITENUM = 0
##output_loc = "decoy_conf.data"
output_loc = "alt-cluster.results"

def seq_len(seq):
    #input: mpair
    #output: len
    seqlen = 0
    for b in seq:
        seqlen += b[0]
        seqlen += b[1]
    return seqlen

def seq_dist(seq1, seq2):
    sseq = set_superseq([seq1, seq2])
    return 2 * seq_len(sseq) - seq_len(seq1) - seq_len(seq2)

def set_superseq(mpairs):
    #given a set of mpairs, find the smallest supersequence
    max_mpairs_len = 0
    for m in mpairs:
        if len(m) > max_mpairs_len:
            max_mpairs_len = len(m)

    superseq = []
    for burst_ind in range(0, max_mpairs_len):
        max_out = 0
        max_inc = 0
        for m_i in range(0, len(mpairs)):
            if len(mpairs[m_i]) > burst_ind:
                max_out = max(mpairs[m_i][burst_ind][0], max_out)
                max_inc = max(mpairs[m_i][burst_ind][1], max_inc)
        superseq.append([max_out, max_inc])
    return superseq

def almost_set_superseq(mpairs, k):
    #given a set of mpairs, find the smallest supersequence
    #except k of them
    #uses a recursive scheme which is only an approximate
    if (k == 0):
        return set_superseq(mpairs)

    #remove the most expensive element
    remove_lengths = []
    for i in range(0, len(mpairs)):
        this_mpairs = list(mpairs)
        this_mpairs.pop(i)
        this_length = seq_len(set_superseq(this_mpairs))
        remove_lengths.append([i, this_length])

    remove_lengths = sorted(remove_lengths, key = lambda r:r[1])
    this_mpairs = list(mpairs)
    this_mpairs.pop(remove_lengths[0][0])
    return almost_set_superseq(this_mpairs, k-1)

try:
    output_loc = str(sys.argv[1])
except:
    print "warning: using default output_loc", output_loc

data = []

origlen = 0
orignum = 0
for i in range(0, CLOSED_SITENUM):
    data.append([])
    for j in range(0, CLOSED_INSTNUM):
        fname = str(i) + "-" + str(j) + ".burst"
        mbursts = read_mbursts(fold + fname)
        mpairs = mbursts_to_mpairs(mbursts)
        data[-1].append(mpairs)
        origlen += seq_len(mpairs)
        orignum += 1

##fout = open(output_loc, "w")
##fout.close()

trial_start = random.randint(0, 399)

for trial in range(trial_start, trial_start + 400):
    print systime.time()
    trial_x = trial/20
    trial_y = trial%20
    characteristic_cover = 0.05 * (trial_x + 1)
    superseq_num = 5 * (trial_y + 1)
    #print trial_x, trial_y

    #for each site, we find one characteristic supersequence
    char_seqs = []
    for i in range(0, len(data)):
        throw_num = int(len(data[i]) * (1- characteristic_cover))
        seq = almost_set_superseq(data[i], throw_num)
        char_seqs.append(seq)
    #char_seqs will change with clustering; it is always len(data[i])

    import numpy
    lens = []
    for c in char_seqs:
        lens.append(seq_len(c))
    #print "Original", float(origlen)/orignum
    #print "Padded", numpy.mean(lens)

    #now, merge those characteristic supersequennces into a few using
    #trivial clustering ("single-linkage clusering" lol)
    #since number is small, inefficient algorithm is ok


    unique_char_seqs = []
    for c in char_seqs:
        if not(c in unique_char_seqs):
            unique_char_seqs.append(c)

    while (len(unique_char_seqs) > superseq_num):
        min_dist = 100000
        min_pair = (0, 0)
        for i in range(0, len(char_seqs)):
            for j in range(0, len(char_seqs)):
                if char_seqs[i] != char_seqs[j]:
                    d = seq_dist(char_seqs[i], char_seqs[j])
                    if (min_dist > d):
                        min_dist = d
                        min_pair = (i, j)

        mergeseq1 = char_seqs[min_pair[0]]
        mergeseq2 = char_seqs[min_pair[1]]
        sseq = set_superseq([mergeseq1, mergeseq2])
        for i in range(0, len(char_seqs)):
            if char_seqs[i] == mergeseq1 or char_seqs[i] == mergeseq2:
                char_seqs[i] = sseq
        unique_char_seqs = []
        for c in char_seqs:
            if not(c in unique_char_seqs):
                unique_char_seqs.append(c)
                
    lens = []
    for c in char_seqs:
        lens.append(seq_len(c))
    #print "Double padded", numpy.mean(lens)

    #now try to send everything under its assigned charseq

    #each charseq has a record of the sites that were -successfully- sent under it

    charseq_successes = []
    for i in range(0, len(unique_char_seqs)):
        charseq_successes.append([])

    undefended = 0
    defended = 0
    padlen = 0
    for i in range(0, CLOSED_SITENUM):
        for j in range(0, CLOSED_INSTNUM):
            if set_superseq([data[i][j], char_seqs[i]]) == char_seqs[i]:
                ind = unique_char_seqs.index(char_seqs[i])
                charseq_successes[ind].append(i)
                padlen += seq_len(char_seqs[i])
                defended += 1
            else:
                undefended += 1
                padlen += seq_len(data[i][j])

    acc_success = cclass_to_acc(charseq_successes)

    acc = (acc_success * defended + undefended)/(defended+undefended)

    fout = open(output_loc, "a+")
    fout.write("Parameters: " + str(characteristic_cover) + " " + \
               str(superseq_num) + "\n")
    fout.write("Overhead: " + str(padlen/float(origlen)) + "\n")
    fout.write("Accuracy: " + str(acc) + "\n")
    fout.write("\n")
    fout.close()

    


