from transform import *

import random, sys

CLOSED_SITENUM = 100
CLOSED_INSTNUM = 80
oh = 0.2

def disturb(size):
    #adds random packets to bursts
    #first divide into bursts
    bursts = []
    last_s = -10000
    for s in size:
        if s != last_s:
            bursts.append([])
        bursts[-1].append(s)
        last_s = s
    actual_oh = random.uniform(oh/2, oh*3/2)

    padsize = []

    for burst_i in range(0, len(bursts)):
        burst = bursts[burst_i]
        this_dir = burst[0]
        padburst = list(burst)
        if burst_i < 6:
            this_oh = random.uniform(0, actual_oh*5)
        else:
            this_oh = random.uniform(0, actual_oh*2)

        for i in range(0, int(this_oh * len(burst))):
            padburst.append(-this_dir)
        random.shuffle(padburst)
        for p in padburst:
            padsize.append(p)

    return padsize

oh_int = random.randint(2, 50)
oh = 1/float(oh_int)

fnames = []
fold = "datasantrim/"
foldout = "defdatanoise/"
for i in range(0, CLOSED_SITENUM):
    for j in range(0, CLOSED_INSTNUM):
        fnames.append(str(i) + "-" + str(j) + ".size")
sizetotal = 0
padsizetotal = 0

for fname in fnames:
    size = read_msizes(fold + fname)
    padsize = disturb(size)
    sizetotal += len(size)
    padsizetotal += len(padsize)
    write_msizes(foldout + fname, padsize)

fout = open("alt-noise.results", "a+")
fout.write("Overhead: " + str(padsizetotal/float(sizetotal)) + "\n")
fout.close()
