#Variation of cluster.py for bw/accuracy.
#Fixes the base time overhead. 

import random
import math
import sys
from split import *
from cclass import *

fold = "datasantrim/"
output_loc = "cluster-fixedtime.results"

#for training
SITE_NUM = 40
INST_NUM = 90
OPENSITE_NUM = 0 #this algorithm can't train with open world

#for testing
TESTSITE_NUM = 60
TESTINST_NUM = 90
TESTOPENSITE_NUM = 5000

sizes = [] #sizes[i] is a list of [-1, 1...]. global
times = [] #as sizes

MIN_TIME = 0.002 #minimum amount of time to wait between packets
MAX_TIME = 0.0075188 #maximum amount of time to wait between packets
#MAX_TIME is set at the 150% time overhead mark

testsizes = [] #used for testing the supersequence learned from sizes
testtimes = [] #as testsizes

def learn_superseq(clus):
    #find the supersequence of clus using a voting algorithm
    #INPUT: clus, a set of indices of web pages. for example sizes[clus[i]] is a packet sequence
    #OUTPUT: the supersequence
    
    pointers = [] #points at each voting packet sequence. incremented if their vote passes

    done = 0 #number of packet sequences that have finished voting
    totaldone = len(clus) #total number of packet sequences

    cur_time = 0
    
    #use all sets that have been loaded:
    for i in range(0, len(clus)):
        pointers.append(0)

    sseq = [] #sizes
    tseq = [] #times
        
    while (done != totaldone):
        votes = []
        #make your votes
        for i in range(0, len(clus)):
            #time: only allow votes within the time frame

            if (pointers[i] != -1): #-1 means this clus finished voting
                if times[clus[i]][pointers[i]] - cur_time < MAX_TIME: #if time is within time range
                    votes.append([sizes[clus[i]][pointers[i]], times[clus[i]][pointers[i]]])
        
        #make a decision for sseq
        vote_size = 0
        for v in votes:
            s = v[0]
            if (s > 0):
                vote_size += 3.5
            else:
                vote_size -= 1
                
        if (vote_size > 0):
            thispacket = 1
            sign = 0
        else:
            thispacket = -1
            sign = 1
        sseq.append(thispacket)

        #make a decision for tseq
        vote_time = cur_time + MIN_TIME
        for v in votes:
            s, t = v[0], v[1]
            if (s * thispacket > 0): #if it is voted
                vote_time = max(t, vote_time)
        tseq.append(vote_time)
        
        #progress pointers        
        for i in range(0, len(clus)):
            if (pointers[i] != -1):
                if (sizes[clus[i]][pointers[i]] * thispacket > 0 and times[clus[i]][pointers[i]] <= vote_time):
                    pointers[i] += 1
                    
            if (pointers[i] == len(sizes[clus[i]])):
                pointers[i] = -1
                done += 1

        #progress time
        cur_time = vote_time

    return sseq, tseq

def learn_steps(superseq, clus):

    #first, how many packets did it cost to transmit everything
    transmit_lens = [] #one for each in clus
    for i in clus:
        #true site is:
        if (i >= SITE_NUM * INST_NUM):
            site = SITE_NUM
        else:
            site = i / INST_NUM
        t_len, t_time = transmit(sizes[i], times[i], ssuperseq, tsuperseq)
        transmit_lens.append([t_len, t_time, site])

    sitecount = [0] * (SITE_NUM + 1)

    transmit_lens = sorted(transmit_lens, key = lambda t:-t[0])


    steps = []

    #use just a regular algorithm

    for i in range(0, STEP_NUM):
        ind = int(len(ssuperseq)/float(STEP_NUM) * i)
        steps.append([ind,tsuperseq[ind]])

    steps = sorted(steps)

    return steps
    


def transmit(s_seq, t_seq, s_superseq, t_superseq):
    #Input: Two sequencees of sizes and times respectively
    #Output: number of packets and time required
    #if superseq is not large enough, we extend it with extendseq

    pt = 0 #points at superseq

    s_bonusseq = [1, -1, -1, -1, -1]
    t_bonusseq = [] #note that this is interpacket time, not actual time
    for i in range(0, 5):
        t_bonusseq.append(t_seq[-i-1] - t_seq[-i-2])
    mys_superseq = list(s_superseq) #may get extended
    myt_superseq = list(t_superseq)
    for i in range(0, len(s_seq)):
        s = s_seq[i]
        t = t_seq[i]
        while (mys_superseq[pt] * s < 0 or t > myt_superseq[pt]):
            pt += 1
            if (pt >= len(mys_superseq)):
                for i in range(0, 5):
                    mys_superseq.append(s_bonusseq[i])
                    myt_superseq.append(myt_superseq[-1] + t_bonusseq[i]) #it's interpacket time
        pt += 1
        if (pt >= len(mys_superseq)):
            for i in range(0, 5):
                mys_superseq.append(s_bonusseq[i])
                myt_superseq.append(myt_superseq[-1] + t_bonusseq[i]) #it's interpacket time
        
    return pt, myt_superseq[pt] - myt_superseq[0]

def pad(steps, transmitlen):
    steps = sorted(steps)
    plength = 0
    ptime = 0
    for step in steps:
        length = step[0]
        t = step[1]
        if length >= transmitlen:
            plength = length
            ptime = t
##            print "normal", plength, ptime
            break

    if (plength == 0): #this means transmitlen > max(steps)
        inc_length = steps[-1][0] * 0.2 + 1
        inc_time = steps[-1][1] * 0.2 + 1
        plength = steps[-1][0]
        ptime = steps[-1][1]
        while (plength < transmitlen):
            plength += inc_length
            ptime += inc_time

    return plength, ptime

def load(sizes, times, fnames):
    for fname in fnames:
        #Set up times, sizes
        f = open(fold + fname + ".cell", "r")
        sizes.append([])
        times.append([])
        for x in f:
            x = x.split("\t")
            sizes[-1].append(int(x[1]))
            times[-1].append(float(x[0]))
        starttime = times[-1][0]
        for i in range(0, len(times[-1])):
            times[-1][i] -= starttime
        f.close()

try:
    output_loc = str(sys.argv[1])
except:
    print "warning: using default output_loc", output_loc

fout = open(output_loc, "w")
fout.close()

#loading data
fnames = []
for site in range(0, SITE_NUM):
    for instance in range(0, INST_NUM):
        fname = str(site) + "-" + str(instance)
        fnames.append(fname)
for site in range(0, OPENSITE_NUM):
    fnames.append(str(site))
load(sizes, times, fnames)
    
testfnames = []
for site in range(0, TESTSITE_NUM):
    for instance in range(0, TESTINST_NUM):
        fname = str(site + SITE_NUM) + "-" + str(instance)
        testfnames.append(fname)
for site in range(0, TESTOPENSITE_NUM):
    testfnames.append(str(site + OPENSITE_NUM))
load(testsizes, testtimes, testfnames)

import time as systime

for trial in range(0, 200):
    print systime.time()
    STEP_NUM = trial * 5 + 5
    STEP_NUM = 1500

    #get supersequence of everything
    clus = []
    for i in range(0, len(sizes)):
        clus.append(i)

    ssuperseq, tsuperseq = learn_superseq(clus) #size, time
    steps = learn_steps(ssuperseq, clus)

    #now test it
    padlentotal = 0
    padtimetotal = 0
    origlentotal = 0
    origtimetotal = 0
    transmitlentotal = 0
    transmittimetotal = 0

    collision_sets = []
    collision_seqs = []
    for i in range(0, len(testsizes)):
        #get true site
        if i >= TESTSITE_NUM * TESTINST_NUM:
            site = TESTSITE_NUM
        else:
            site = i/TESTINST_NUM
            
        transmitlen, transmittime = transmit(testsizes[i], testtimes[i], ssuperseq, tsuperseq)
        padlen, padtime = pad(steps, transmitlen)

        #populate the collision class
        if (padlen in collision_seqs):
            ind = collision_seqs.index(padlen)
            collision_sets[ind].append(site)
        else:
            collision_sets.append([site])
            collision_seqs.append(padlen)

        #calculate overheads        
        padlentotal += padlen
        padtimetotal += padtime
        transmitlentotal += transmitlen
        transmittimetotal += transmittime
        origlentotal += len(testsizes[i])
        origtimetotal += testtimes[i][-1] - testtimes[i][0]

    #calculate accuracy from collision sets
    accuracy = cclass_to_acc(collision_sets)

    fout = open(output_loc, "a+")
##    fout.write("Parameters: " + str(steps) + "\n")
    fout.write("Transmit Bandwidth overhead: " + str(transmitlentotal/float(origlentotal)) + "\n")
    fout.write("Transmit Time overhead: " + str(transmittimetotal/origtimetotal) + "\n")
    fout.write("Bandwidth overhead: " + str(padlentotal/float(origlentotal)) + "\n")
    fout.write("Time overhead: " + str(padtimetotal/origtimetotal) + "\n")
    fout.write("Accuracy: " + str(accuracy) + "\n")
    fout.write("\n")
    fout.close()




