import scipy
import numpy
import scipy.stats
import random
import math
import time

from padding_prob import *

def get_combs_filtered(num1, num2, possibles):
    #get all possible sequences of length num2 where the number of 0's is num1
    #possibles is a len(num2) list
    #each possible = [positions where this thing can be] (index in num2), -1 is fake
    #possibles is a boolean list of length (num2), possibles[i] is (can seq[i] be 0?)
    #num1 is small, num1 and num2 may be large
    #recursion is done on num2; one num2 is removed at a time

    fakecount = 0

    for p in possibles:
        if -1 in p:
            fakecount += 1

    if (num1 > fakecount):
        return -1 #impossible

    if (num1 == num2):
        if (num1 == fakecount):
            return [[0] * num1] #also deals with the terminal case
        else:
            return -1

    if (num2 == 0 and num1 != 0):
        return -1

    combs = []

    #decrement all possibles by 1
    new_num2 = num2 - 1

    #two possibilities: real and fake
##    print possibles
    if 0 in possibles[0]: #real
        new_num1 = num1
        #decrement the possibles indices
        new_possibles = []
        for p in possibles[1:]:
            new_p = []
            for num in p:
                if (num == -1):
                    new_p.append(num)
                elif (num > 0):
                    new_p.append(num-1)
            new_possibles.append(new_p)
        new_combs = get_combs_filtered(new_num1, new_num2, new_possibles)
        if (new_combs != -1):
            for comb in new_combs:
                combs.append([1] + comb)
    if -1 in possibles[0]: #fake
        new_num1 = num1 - 1
        #don't need to decrement possibles indices
        new_possibles = possibles[1:]
        new_combs = get_combs_filtered(new_num1, new_num2, new_possibles)
        if (new_combs != -1):
            for comb in new_combs:
                combs.append([0] + comb)
    if len(combs) == 0:
        return -1

    return combs
    

class equal_defense():
    def __init__(self):
        self.real_inc_add = []
        self.real_out_add = []
        self.fake_inc_add = []
        self.fake_out_add = []
        self.fake_max = 0
        self.fake_prob = 0

    def read_file(self, fname):
        func = {"real_inc_add": [],
                "real_out_add": [],
                "fake_inc_add": [],
                "fake_out_add": [],
                "fake_max": [],
                "fake_prob": []}

        
        f = open(fname, "r")
        lines = f.readlines()
        f.close()

        for li in lines:
            if (li != "\n"):
                li = li.split("\t")
                li_type = li[0].strip()
                if (li_type in ["fake_max"]):
                    func[li_type] = int(li[1])
                elif (li_type in ["fake_prob"]):
                    func[li_type] = float(li[1])
                else:
                    dType = str(li[1])
                    pmin = float(li[3])
                    pmax = float(li[4])
                    param = parse_list(li[2])
                    d = distr(dType, param, pmin, pmax)
                    func[li_type].append(d)

        if (func["real_inc_add"] != []):
            self.real_inc_add = func["real_inc_add"]
        if (func["real_out_add"] != []):
            self.real_out_add = func["real_out_add"]
        if (func["fake_inc_add"] != []):
            self.fake_inc_add = func["fake_inc_add"]
        if (func["fake_out_add"] != []):
            self.fake_out_add = func["fake_out_add"]
        if (func["fake_max"] != []):
            self.fake_max = func["fake_max"]
        if (func["fake_prob"] != []):
            self.fake_prob = func["fake_prob"]

    def load_example(self):
        for i in range(0, 100):
            self.real_inc_add.append(distr("uniform", [0, 200], 0, 210))
            self.real_out_add.append(distr("uniform", [0, 200], 0, 210))
            self.fake_inc_add.append(distr("uniform", [1, 200], 0, 210))
            self.fake_out_add.append(distr("uniform", [1, 200], 0, 210))
        self.fake_max = 5
##        self.fake_prob = 0.3

    def write_file(self, fname):
        fout = open(fname, "w")
        for r in self.real_inc_add:
            fout.write("real_inc_add" + "\t" + repr(r) + "\n")
        for r in self.real_out_add:
            fout.write("real_out_add" + "\t" + repr(r) + "\n")
        for r in self.fake_inc_add:
            fout.write("fake_inc_add" + "\t" + repr(r) + "\n")
        for r in self.fake_out_add:
            fout.write("fake_out_add" + "\t" + repr(r) + "\n")
        fout.write("fake_max\t" + repr(self.fake_max) + "\n")
        fout.write("fake_prob\t" + repr(self.fake_prob) + "\n")
        fout.close()

    def pad(self, sequence):
##        fake_num = random.randint(1, self.fake_max)
##        target_burst_num = fake_num + len(sequence)

        #get a real/fake marker
##        real_is = []
##        for i in range(0, target_burst_num):
##            real_is.append(i)
##        real_is = random.sample(real_is, len(sequence))
##
##        burst_real = []
##        for i in range(0, target_burst_num):
##            if i in real_is:
##                burst_real.append(1)
##            else:
##                burst_real.append(0)
##
##        real_pointer = 0 #points at real_inc and real_out
        #note that we don't use a fake pointer

        fake_num = 0

        padsequence = []
        fake_pointer = 0
        for i in range(0, len(sequence)):
            r = random.uniform(0, 1)
            while r < self.fake_prob and fake_num < self.fake_max:
                #add a fake burst
                out = int(self.fake_out_add[len(padsequence)].draw())
                inc = int(self.fake_inc_add[len(padsequence)].draw())
                padsequence.append([out, inc])
                r = random.uniform(0, 1)
                fake_num += 1
                
            out = int(self.real_out_add[i].draw()) + sequence[i][0]
            inc = int(self.real_inc_add[i].draw()) + sequence[i][1]
            padsequence.append([out, inc])

        return padsequence

    def logprob_pad(self, pad_sequence, sequence):
        logp = 0

        fake_num = len(pad_sequence) - len(sequence)
        if fake_num < 0 or fake_num > self.fake_max:
            return 1

        #build up the possibles list
        #possibles[i] is where pad_sequence[i] could be
        #also build fake_prob here just to save time

        possibles = []
        fake_prob = []
        for pseq_i in range(0, len(pad_sequence)):
            possible = []
            #real:
            for seq_i in range(0, len(sequence)):
                out_add = pad_sequence[pseq_i][0] - sequence[seq_i][0]
                inc_add = pad_sequence[pseq_i][1] - sequence[seq_i][1]
                out_prob = self.real_out_add[seq_i].prob(out_add, out_add + 1)
##                print out_add, out_prob
                inc_prob = self.real_inc_add[seq_i].prob(inc_add, inc_add + 1)
##                print inc_add, inc_prob
                if inc_prob * out_prob > 0:
                    possible.append(seq_i)

            #fake:
            out_add = pad_sequence[pseq_i][0]
            out_prob = self.fake_out_add[pseq_i].prob(out_add, out_add+1)
            inc_add = pad_sequence[pseq_i][1]
            inc_prob = self.fake_inc_add[pseq_i].prob(inc_add, inc_add+1)
            if inc_prob * out_prob > 0:
                possible.append(-1)
            fake_prob.append(inc_prob * out_prob)
            
            possibles.append(possible)

##        print pad_sequence
##        print sequence
##        print possibles
            

##        fake_possibles = []
##        fake_prob = []
##        for b in range(0, target_burst_num):
##            out_add = pad_sequence[b][0]
##            out_prob = self.fake_out_add[b].prob(out_add, out_add+1)
##            inc_add = pad_sequence[b][1]
##            inc_prob = self.fake_inc_add[b].prob(inc_add, inc_add+1)
##            if inc_prob * out_prob > 0:
##                fake_possibles.append(1)
##                fake_prob.append(inc_prob * out_prob)
##            else:
##                fake_possibles.append(0)
##                fake_prob.append(0)


        #optimization: filter out combs where fake is not possible
        #enumerate over all possible combinations of reals and fakes
        combs = get_combs_filtered(len(pad_sequence) - len(sequence),
                                   len(pad_sequence), possibles)

        if combs == -1:
            return 1

        #precompute real_prob to minimize the number of distr().prob calls

        real_prob = []

        for b_seq in range(0, len(sequence)):
            this_prob = []
            for b_pad in range(0, len(pad_sequence)):
                pad_out = pad_sequence[b_pad][0]
                pad_inc = pad_sequence[b_pad][1]
                
                out = sequence[b_seq][0]
                inc = sequence[b_seq][1]

                inc_add = pad_inc - inc
                inc_prob = self.real_inc_add[b_seq].prob(inc_add, inc_add+1)
                out_add = pad_out - out
                out_prob = self.real_out_add[b_seq].prob(out_add, out_add+1)
                this_prob.append(inc_prob * out_prob)

            real_prob.append(this_prob)

##        print real_prob
        
        target_burst_num = len(pad_sequence)
        
        #note that each comb is exactly equally likely
        
        #test if each one of them is possible
        p = 0
        for burst_real in combs:
            burst_p = 1
            real_pointer = 0 #points at sequence/real_*_add
            #no fake pointer!
            for b in range(0, target_burst_num): #b points at pad_sequence
                if (burst_real[b] == 1):
                    prob = real_prob[real_pointer][b]
                    real_pointer += 1
                if (burst_real[b] == 0):
                    prob = fake_prob[b]

                burst_p *= prob

            p += burst_p

        p *= math.pow(self.fake_prob, fake_num)
        p *= math.pow(1 - self.fake_prob, len(sequence))

        if p == 0:
            return 1
        else:
            return math.log(p)

    def optimize_add(self, cost, mpairs_data):
        #strategy: add cost% of the original data

        add_nums = [] #add_nums[i] is a list of [out, inc] pairs for burst i

        for b_i in range(0, 100):
            add_nums.append([])

        for site in mpairs_data:
            for seq in site:
                for b_i in range(0, len(seq)):
                    add_nums[b_i].append(seq[b_i])

        default_inc = self.real_inc_add[0]
        default_out = self.real_out_add[0]
        for b_i in range(0, 100):
            if len(add_nums[b_i]) > 10:
                for oi in range(0, 2):
                    s = 0
                    t = 0
                    for a in add_nums[b_i]:
                        s += a[oi]
                        t += 1
                    s /= float(t)
                    d = distr("uniform", [0, int(s * 2 * cost) + 2], 0, int(s * 2 * cost) + 2)
                    if (oi == 0):
                        self.real_out_add[b_i] = d
                        default_out = d
                    if (oi == 1):
                        self.real_inc_add[b_i] = d
                        default_inc = d
            else:
                self.real_out_add[b_i] = default_out
                self.real_inc_add[b_i] = default_inc
                

    def optimize_fake(self, cost, mpairs_data, dType = "normal", force_equal = 0):
        #strategy: simulate real traffic, but take cheapest percentage of real traffic

        #cost >=0, <1 is percentage of traffic to consider
        #higher = higher cost

        #repeat number of times according to practical time limits...

        fake_nums = []
        for i in range(0, 100):
            fake_nums.append([])
        #fake_nums[i] is a list of [out, inc] pairs for burst i

        CLOSED_SITENUM = len(mpairs_data) - 1

        for trial in range(0, 1000):
            picked = 0
            while (picked == 0):
                #pick a random pair in the monitored set
                site1 = random.randint(0, CLOSED_SITENUM-1)
                site2 = (site1 + random.randint(1, CLOSED_SITENUM-1)) % CLOSED_SITENUM
                seq1 = random.choice(mpairs_data[site1])
                seq2 = random.choice(mpairs_data[site2])

                if abs(len(seq1) - len(seq2)) <= self.fake_max:
                    picked = 1

            for blen in range(max(len(seq1), len(seq2)), min(len(seq1), len(seq2)) + self.fake_max + 1):
                #fake locations are random, get a few
                locations = []

                for b_i in range(0, blen):
                    locations.append(b_i)

                for sample_i in range(0, 20):
                    reals1 = random.sample(locations, len(seq1))
                    reals2 = random.sample(locations, len(seq2))

                    reals1 = sorted(reals1)
                    reals2 = sorted(reals2)

                    real_pointer1 = 0
                    real_pointer2 = 0

                    for b_i in range(0, blen):
                        if b_i in reals1 and not(b_i in reals2):
                            seq = list(seq1[real_pointer1])
                            seq[0] += self.real_out_add[real_pointer1].draw()
                            seq[1] += self.real_inc_add[real_pointer1].draw()
                            fake_nums[b_i].append(seq)
                        if b_i in reals2 and not(b_i in reals1):
                            seq = list(seq2[real_pointer2])
                            seq[0] += self.real_out_add[real_pointer2].draw()
                            seq[1] += self.real_inc_add[real_pointer2].draw()
                            fake_nums[b_i].append(seq)
                        if b_i in reals1:
                            real_pointer1 += 1
                        if b_i in reals2:
                            real_pointer2 += 1

        default_fake_out = self.fake_out_add[0]
        default_fake_inc = self.fake_inc_add[0]

        for i in range(0, 100):
            if (len(fake_nums[i]) > 10):
                for oi in range(0, 2):
                    packets = []
                    for f in fake_nums[i]:
                        packets.append(f[oi])

                    packets = sorted(packets)
                    le = len(packets)
                    packets = packets[le/20:(19*le)/20]
                    le = len(packets)
                    packets = packets[:int(cost*le)]
                    mi = packets[0]
                    ma = packets[-1] + 2
                    if (force_equal == 1):
                        if (oi == 0):
                            ma = mi + self.real_out_add[i].param[1] + 2 #the maximum of the adding range
                        else:
                            ma = mi + self.real_inc_add[i].param[1] + 2
                        d = distr("uniform", [mi, ma], mi, ma)

                    else:
                        d = distr()
                        d.fit(dType, mi, ma, packets)
                    
                    if (oi == 0):
                        self.fake_out_add[i] = d
                        default_fake_out = d
                    else:
                        self.fake_inc_add[i] = d
                        default_fake_inc = d
            else:
                self.fake_out_add[i] = default_fake_out
                self.fake_inc_add[i] = default_fake_inc

    def find_steps_oh(self, max_oh, nums):
        #finds steps like below, but based on overhead
        steps_results = []
        nums = sorted(nums)
        for prob_trial in range(0, 500):
            rise_probs = []
            STEPNUM = random.randint(2, int(max(nums)))
            for i in range(0, STEPNUM):
                rise_probs.append(random.uniform(0, 1))

            s = sum(rise_probs)
            for i in range(0, len(rise_probs)):
                rise_probs[i] /= s

            rise_probs = sorted(rise_probs)
            rise_probs.reverse()

            #from rise_probs, derive steps
            steps = []
            totalprob = 0
            for prob in rise_probs:
                totalprob += prob
                ind = int(len(nums) * totalprob)
                if ind == len(nums):
                    ind -= 1
                steps.append(nums[ind])

            #cut down on steps
            uniq_steps = []
            for s in steps:
                if not(s in uniq_steps):
                    uniq_steps.append(s)
            steps = uniq_steps

            stepped_nums = [[]] #stepped_nums[i] is a rise

            current_step_i = 0

            for n_i in range(0, len(nums)):
                if nums[n_i] > steps[current_step_i]:
                    current_step_i += 1
                    stepped_nums.append([])
                stepped_nums[-1].append(nums[n_i])

            oh = 0
            total = 0
            for rise in stepped_nums:
                m = max(rise)
                for num in rise:
                    oh += m - num
                    total += num

            oh = oh/float(total)

            cover = 0
            total_nums = len(nums)
            for rise in stepped_nums:
                prob = len(rise)/float(total_nums)
                cover += prob * prob

            if (oh < max_oh):
                steps_results.append([steps, cover])

        if (steps_results == []):
            #return all nums
            steps = []
            for n in nums:
                if not(n in steps):
                    steps.append(n)
            steps = sorted(steps)
            return steps

        steps_results = sorted(steps_results, key = lambda steps_results: -steps_results[1])
        
        steps = steps_results[0][0]

        return steps
            
    
    def optimize_burst(self, max_oh, mpairs_data):
        #optimizes self.burst_steps
        
        burst_nums = []
        CLOSED_SITENUM = len(mpairs_data) - 1
        CLOSED_INSTNUM = len(mpairs_data[0])
        for i in range(0, CLOSED_SITENUM):
            for j in range(0, CLOSED_INSTNUM):
                seq = mpairs_data[i][j]
                burst_nums.append(len(seq))

        learned_steps = self.find_steps_oh(max_oh, burst_nums)

        self.burst_steps = []
        for l in learned_steps:
            if (l <= 50):
                self.burst_steps.append(l)
                    
        
##d = equal_defense()
##d.load_example()
##
##data = []
##
##for i in range(0, 100):
##    randseq = []
##    for j in range(0, 13):
##        randburst = [random.randint(1, 10), random.randint(1, 300)]
##        randseq.append(randburst)
##    data.append(randseq)
##
##pad_seq = d.pad(data[0])
##
##
##a = time.time()
##
##for i in range(0, 100):
##    d.logprob_pad(pad_seq, data[i])
##
##b = time.time()
##print b - a
                

        
