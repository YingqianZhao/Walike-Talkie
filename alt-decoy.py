import random

from transform import *
fold = "datasantrim/"
CLOSED_SITENUM = 100
CLOSED_INSTNUM = 90
OPEN_SITENUM = 0
output_loc = "alt-decoy.results"

def seq_len(seq):
    #returns the number of packets in a mbursts sequence
    #analogous to len() for cells
    total_len = 0
    for s in seq:
        total_len += s[0]
        total_len += s[1]
    return total_len

def is_superseq(a, b):
    #INPUT: two mpairs format
    #OUTPUT: 1 if b is a superseq of a
    if (len(b) < len(a)):
        return 0

    a_ptr = 0
    b_ptr = 0
    while b_ptr < len(b) and a_ptr < len(a):
        if a[a_ptr][0] <= b[b_ptr][0] + 2 and \
           a[a_ptr][1] <= b[b_ptr][1]:
            a_ptr += 1
        b_ptr += 1

    if b_ptr == len(b): #we finished b without sending it
        return 0
    else:
        return 1
    

#load sequences
data = []
for i in range(0, CLOSED_SITENUM):
    data.append([])
    for j in range(0, CLOSED_INSTNUM):
        fname = str(i) + "-" + str(j) + ".burst"
        mbursts = read_mbursts(fold + fname)
        data[-1].append(mbursts_to_mpairs(mbursts))

fout = open(output_loc, "w")
fout.close()

#split sequences into decoy and nondecoy

pop = []
for i in range(0, CLOSED_SITENUM):
    pop.append(i)
        
for decoy_trial in range(1, 10000):
    decoy_num = random.randint(2, 20)
    decoy_sites = random.sample(pop, decoy_num)
    
    decoy_data = []
    nondecoy_data = []
    for i in range(0, len(data)): #sites
        if (i in decoy_sites):
            decoy_data.append(data[i])
        else:
            nondecoy_data.append(data[i])

    #for each nondecoy packet sequence, find the closest decoy packet sequence

    ##decoy_site_counts = [0] * len(decoy_sites) # no. of times each site was used as a decoy
    total_nonseq = 0 #total number of non_seqs
    total_cost = 0
    undefended = 0

    dresults = []
    #each element k is the best defense of a nonseq by a given site
    #k[0] is the chance of success
    #k[1] is the site chosen
    #k[2] is the cost

    for non_i in range(0, len(nondecoy_data)):
        for non_seq_i in range(0, len(nondecoy_data[non_i])):
            #compute all decoy packet sequence costs
            non_seq = nondecoy_data[non_i][non_seq_i]

            #disallow really short seuqneces     
            if (seq_len(non_seq) < 20):
                continue
            
            decoy_chances = [] #chance of using each site to decoy this non_seq
            decoy_costs = [] #cost of using this site
            for decoy_i in range(0, len(decoy_data)):
                decoy_chance = 0 #decoy chance of this site
                decoy_cost = 0 #decoy cost of using this site on average
                for decoy_seq_i in range(0, len(decoy_data[decoy_i])):
                    decoy_seq = decoy_data[decoy_i][decoy_seq_i]
                    
                    #disallow realy short decoy sequences
                    if (seq_len(decoy_seq) < 20):
                        continue
                    
                    can_decoy = is_superseq(decoy_seq, non_seq)
                    decoy_chance += can_decoy
                    if (can_decoy == 1):
                        decoy_cost += seq_len(non_seq) - seq_len(decoy_seq)
                if (decoy_chance) > 0:
                    decoy_cost /= float(decoy_chance) #avg decoy cost of only successful cases
                else:
                    decoy_cost = -1
                decoy_chance /= float(len(decoy_data[decoy_i]))
                decoy_chances.append(decoy_chance)
                decoy_costs.append(decoy_cost)

            #candidate site is the one with the highest decoy chance

            dchance = max(decoy_chances)
            dchance_ind = decoy_chances.index(dchance)
            dsite = dchance_ind
            dcost = decoy_costs[dchance_ind]

            if (dchance > 0):
                dresults.append([dchance, dsite, dcost])
            else:
                undefended += 1
            total_nonseq += 1

    bdr = 0.3  #chance of visiting decoy set, in total

    #calculate each site's number of responsible sequences from dresults
    decoy_site_numseq = [0] * len(decoy_sites)
    for dresult in dresults:
        decoy_site_numseq[dresult[1]] += 1

    #for each nonseq that can be decoyed, calculate chance of it being a decoy
    #then compare with chance of it being real to get the accuracy of the attacker
    accs = []
    for dresult in dresults:
        dprob = dresult[0]
        dprob *= 1.0/len(decoy_sites)
        dprob *= bdr
        dprob *= 1.0/decoy_site_numseq[dresult[1]]

        rprob = 1.0/total_nonseq * (1-bdr)

##        accs.append(rprob/(dprob+rprob))

        if (dprob/(dprob+rprob) < 0.1):
            accs.append(1.0)
        else:
            accs.append(0.0)

    ##acc = 0
    ##for dresult in dresults:
    ##    acc += dresult[0]
    ##
    ##acc /= len(dresults)
    ##print acc

    site_decoy_cost = [0] * len(decoy_sites) #how much decoying we expect if this site was called
    for dresult in dresults:
        site_decoy_cost[dresult[1]] += dresult[0] * dresult[2]

    for i in range(0, len(site_decoy_cost)):
        if (decoy_site_numseq[i] != 0): #if it's 0 it's still accurate
            site_decoy_cost[i] /= float(decoy_site_numseq[i])

    avg_cost = sum(site_decoy_cost)/float(len(site_decoy_cost))
    #avgcost is the cost we can expect if we call something in decoy

    avg_nonlen = 0 #total length of nondecoy seqs
    num_nonlen = 0
    for site in nondecoy_data:
        for seq in site:
            avg_nonlen += seq_len(seq)
            num_nonlen += 1
    avg_nonlen /= float(num_nonlen)

    avg_decoylen = 0
    num_decoylen = 0
    for site in decoy_data:
        for seq in site:
            avg_decoylen += seq_len(seq)
            num_decoylen += 1
    avg_decoylen /= float(num_decoylen)

    #total_decoylen + total_cost would now be the total length
    #of packet sequences in decoy_data after decoying

    before_decoy_len =  avg_nonlen * (1 - bdr)
    before_decoy_len += avg_decoylen * bdr
    after_decoy_len  =  avg_nonlen * (1 - bdr)
    after_decoy_len  += (avg_decoylen + avg_cost) * bdr

    oh = after_decoy_len/float(before_decoy_len)
    fout = open(output_loc, "a+")
##    fout.write(str(bdr))
##    fout.write("\t")
    fout.write(str(oh))
    fout.write("\t")
    fout.write(str(sum(accs)/len(accs)))
    fout.write("\n")
    fout.close()
##    print bdr
##    print undefended, total_nonseq
##    print sum(accs)/len(accs)
##    print oh






        
