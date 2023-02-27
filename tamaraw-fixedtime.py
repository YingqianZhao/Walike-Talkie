#Version of tamaraw.py that only looks at time and bandwidth
import math
import random
from cclass import *

DATASIZE = 1
interpacket_times = [0.00826, 0.00510]
padL = 100
output_loc = "tamaraw-fixedtime.results"

CLOSED_SITENUM = 100
CLOSED_INSTNUM = 90
OPEN_SITENUM = 7000

tardist = [[], []]
defpackets = []

def fsign(num):
    if num > 0:
        return 0
    else:
        return 1

def rsign(num):
    if num == 0:
        return 1
    else:
        return abs(num)/num

def get_sizetime(packets):
    size = 0
    for p in packets:
        size += abs(p[1])
    time = packets[-1][0] - packets[0][0]
    return size, time

def get_outinc(packets):
    out_count = 0
    inc_count = 0
    for p in packets:
        if p[1] > 0:
            out_count += 1
        else:
            inc_count += 1
    return out_count, inc_count

##def AnoaTime(parameters):
##    direction = parameters[0] #0 out, 1 in
##    method = parameters[1]
##    if (method == 0):
##        if direction == 0:
##            return 0.04
##        if direction == 1:
##            return 0.012
        

def AnoaPad(list1, list2, padL, method):
    lengths = [0, 0]
    times = [0, 0]
    for x in list1:
        if (x[1] > 0):
            lengths[0] += 1
            times[0] = x[0]
        else:
            lengths[1] += 1
            times[1] = x[0]
        list2.append(x)

##    print times[0], times[1]
    for j in range(0, 2): #j=0 outgoing, j=1 incoming
        curtime = times[j]
        topad = -int(math.log(random.uniform(0.00001, 1), 2) - 1) #1/2 1, 1/4 2, 1/8 3, ...
        if (method == 0):
            topad = (lengths[j]/padL + topad) * padL
        while (lengths[j] < topad):
            curtime += interpacket_times[j]
            if j == 0:
                list2.append([curtime, DATASIZE])
            else:
                list2.append([curtime, -DATASIZE])
            lengths[j] += 1

def Anoa(list1, list2): #inputpacket, outputpacket, parameters
    #Does NOT do padding, because ambiguity set analysis. 
    starttime = list1[0][0]
    times = [starttime, starttime] #lastpostime, lastnegtime
    curtime = starttime
    lengths = [0, 0]
    datasize = DATASIZE
##    parameters[0] = "Constant packet rate: " + str(interpacket_times[0]) + ", "
##    parameters[0] += str(interpacket_times[1]) + ". "
##    parameters[0] += "Data size: " + str(datasize) + ". "
    listind = 0 #marks the next packet to send
    while (listind < len(list1)):
        #decide which packet to send
        if times[0] + interpacket_times[0] < times[1] + interpacket_times[1]:
            cursign = 0
        else:
            cursign = 1
        times[cursign] += interpacket_times[cursign]
        curtime = times[cursign]
        
        tosend = datasize
        while (list1[listind][0] <= curtime and fsign(list1[listind][1]) == cursign and tosend > 0):
            if (tosend >= abs(list1[listind][1])):
                tosend -= abs(list1[listind][1])
                listind += 1
            else:
                list1[listind][1] = (abs(list1[listind][1]) - tosend) * rsign(list1[listind][1])
                tosend = 0
            if (listind >= len(list1)):
                break
        if cursign == 0:
            list2.append([curtime, datasize])
        else:
            list2.append([curtime, -datasize])
        lengths[cursign] += 1

import sys
##import os

try:
    output_loc = str(sys.argv[1])
except:
    print "warning: using default output_loc ", output_loc

fold = "datasantrim/" #takes in .cell format from this folder

##if not os.path.exists(foldout):
##    os.makedirs(foldout)

fnames = []
for site in range(0, CLOSED_SITENUM):
    for inst in range(0, CLOSED_INSTNUM):
        fnames.append(str(site) + "-" + str(inst))

for site in range(0, OPEN_SITENUM):
    fnames.append(str(site))

#all analyzing stuff

fout = open(output_loc, "w")
fout.close()

import time as systime

for trial in range(0, 50):
    print systime.time()
    origsize = 0
    origtime = 0
    padsize = 0
    padtime = 0
    cell_sites = []
    cell_labels = []

    padL = (2 * trial + 2) * 10

    trial_count = 0
    print_trial = 0

    for fname in fnames:
##        trial_count += 1
##        if trial_count > len(fnames)/100 * print_trial:
##            print_trial += 1
##            print trial_count
        packets = []
        with open(fold + fname + ".cell", "r") as f:
            lines = f.readlines()
            starttime = float(lines[0].split("\t")[0])
            for x in lines:
                x = x.split("\t")
                packets.append([float(x[0]) - starttime, int(x[1])])

        s, t = get_sizetime(packets)
        origsize += s
        origtime += t

        #anoa: transform packets to constant-rate packet delivery
        list2 = []
        Anoa(packets, list2)
        list2 = sorted(list2, key = lambda list2: list2[0])

        #anoapad: add packets to constant multiple
        list3 = []
        AnoaPad(list2, list3, padL, 0)
        list3 = sorted(list3, key = lambda list3:list3[0])

        s, t = get_sizetime(list3)
        padsize += s
        padtime += t
        
        if "-" in fname: #closed world
            site = int(fname.split("-")[0])
        else: #open world
            site = CLOSED_SITENUM

        pad_inc_count = 0
        pad_out_count = 0
        for i in range(0, len(list3)):
            if (list3[i][1] > 0):
                pad_out_count += 1
            if (list3[i][1] < 0):
                pad_inc_count += 1
        label = [pad_out_count, pad_inc_count]
        if label in cell_labels:
            cell_sites[cell_labels.index(label)].append(site)
        else:
            cell_sites.append([site])
            cell_labels.append(label)

    accuracy = cclass_to_acc(cell_sites)

    fout = open(output_loc, "a+")
    fout.write("Parameters: " + str(interpacket_times[0]) + " " + \
               str(interpacket_times[1]) + " " + \
               str(padL) + "\n")
    fout.write("Bandwidth overhead: " + str(padsize/float(origsize)) + "\n")
    fout.write("Time overhead: " + str(padtime/origtime) + "\n")
    fout.write("Accuracy: " + str(accuracy) + "\n")
    fout.write("\n")
    fout.close()

