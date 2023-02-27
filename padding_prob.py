#distr() object d:
#d(l) returns a padded num p_l > l
#d allows d.prob(a, b) call
#which is the probability that d(k) lies between a and b
#the defense() module is unused; replaced by equal_padding_prob.py
#distr() is used by equal_padding_prob.py


import scipy
import numpy
import scipy.stats
import random
import math
import time

from split import *



def iprint(string, is_printing):
    if (is_printing == 1):
        print string

class distr():
    def __init__(self, dType = "", param = [], pmin = 0, pmax = 100):
        #dType is a string telling us what type we want
        #parameters determine the exact dType
        self.dType = dType.lower()
        self.setparam(param)
        self.pmin = pmin
        self.pmax = pmax

        if not(self.dType in ["", "normal", "uniform", "kde", "discrete"]):
            print "distr: dType unknown", self.dType
            sys.exit(0)

        if self.pmin > self.pmax:
            print "distr: distr.pmin > distr.pmax"
            sys.exit(0)

        #normal: param[0] is mean, param[1] is sd

        #uniform: param[0] is lower bound, param[1] is upper bound
        #still subject to pmin, pmax

        #kde: param is list of support points

        #discrete: param is a list of pairs [, probability]
        #we don't check sum and limits so the caller should (or this makes no sense)

    def setparam(self, param):
        if self.dType == "kde":
            if len(param) > 100:
                self.param = random.sample(param, 100)
                #we keep some stuff here just for kde to save time
            else:
                self.param = param
            self.kde = scipy.stats.gaussian_kde(self.param)
            self.limit_prob = float(self.kde.integrate_box_1d(self.pmin, self.pmax))

        else:
            self.param = param
        

    def prob(self, a, b):
        #returns probability of a draw coming from (a, b)
        #have a, b conform to limits:
        a = max(self.pmin, a)
        b = min(self.pmax, b)

        if (a >= b):
            return 0 #this happens if (a, b) is out of the range
        
        if self.dType == "uniform":
            dist_min = max(self.pmin, self.param[0])
            dist_max = min(self.pmax, self.param[1])
            dist_range = dist_max - dist_min
            if (dist_range == 0): #dirichlet 
                if (a <= dist_min and b >= dist_max):
                    return 1
                else:
                    return 0
            my_range = b - a
            prob = my_range/float(dist_range)
            
        if self.dType == "normal":
            m = self.param[0]
            s = self.param[1]
            prob = 0.5 * (scipy.special.erf((b - m)/(s * math.sqrt(2))) - \
                          scipy.special.erf((a - m)/(s * math.sqrt(2))))
##            prob = s.cdf(b) - s.cdf(a)

            #now, increase prob based on min/max limits
##            limit_prob = s.cdf(self.pmax) - s.cdf(self.pmin) #prob in limits

            
            limit_prob = 0.5 * (scipy.special.erf((self.pmax - m)/(s * math.sqrt(2))) - \
                                scipy.special.erf((self.pmin - m)/(s * math.sqrt(2))))

            prob = prob / limit_prob

        if self.dType == "kde":
            prob = float(self.kde.integrate_box_1d(a, b))
            prob = prob / self.limit_prob

        if self.dType == "discrete":
            prob = 0
            limit_prob = 0
            for pair in self.param:
                num = pair[0]
                this_prob = pair[1]
                if num >= a and num < b:
                    prob += this_prob
                if num >= self.pmin and num < self.pmax:
                    limit_prob += this_prob
            prob = prob / limit_prob
            
            
        return prob

    def pad_prob(self, p_le, le):
        
        #returns probability of call returning p_le when fed le
        if (p_le < le):
            return 0
        if (le == 0):
            if (p_le == 0):
                return 1
            else:
                return 0
        else:
            range_min = p_le/float(le)
            range_max = (p_le + 1)/float(le)
        return self.prob(range_min - 1, range_max - 1) #-1 because prob takes in a

    def draw(self):
        #draws one sample from the distribution
        if self.dType == "uniform":
            dist_min = max(self.pmin, self.param[0])
            dist_max = min(self.pmax, self.param[1])
            p = random.uniform(dist_min, dist_max)
            
        if self.dType == "normal":
            p = numpy.random.normal(self.param[0], self.param[1], 1)[0]
            drawn = 0
            while (p < self.pmin or p > self.pmax) and drawn < 100:
                p = numpy.random.normal(self.param[0], self.param[1], 1)[0]
                drawn += 1
            if drawn == 100:
                print "Error: Couldn't draw from normal"
                sys.exit(0)

        if self.dType == "kde":
            kde = self.kde
            p = float(kde.resample(1))
            drawn = 0
            while (p < self.pmin or p > self.pmax) and drawn < 100:
                p = float(kde.resample(1))
                drawn += 1
            if drawn == 100:
                print "Error: Cannot draw from kde"
                sys.exit(0)

        if self.dType == "discrete":
            rand = random.uniform(0, 1)
            p = self.param[-1][0]
            for pair in self.param:
                num = pair[0]
                this_prob = pair[1]
                rand -= this_prob
                if (rand <= 0):
                    p = num
                    break
        return p

    def fit(self, dType, pmin, pmax, points):
        self.dType = dType
        self.pmin = pmin
        self.pmax = pmax

        if dType == "normal":
            mean, sd = scipy.stats.norm.fit(points)
            if mean < pmin:
                mean = pmin
            if mean > pmax:
                mean = pmax
            sd = max(sd, mean/10.0)
            self.setparam([mean, sd])
            
        elif dType == "uniform":
            cover_num = 0.7 * len(points)

            points = sorted(points)
            start_point = max(points[0], pmin)
            end_point = min(points[int(0.7 * len(points))], pmax)

            self.setparam([start_point, end_point])
            
        elif dType == "kde":
            self.setparam(points)
            
        else:
            print "fit: cannot understand dType", dType
            sys.exit(0)

    def __repr__(self):
        retstr = ""
        retstr += str(self.dType) + "\t"
        retstr += repr(self.param) + "\t"
        retstr += repr(self.pmin) + "\t"
        retstr += repr(self.pmax)
        return retstr

def parse_list(string):
    #parses num, [num, num], or [[num, num], [num, num]], etc.
    
    if (not("[" in string)): #base case
        return float(string)
    retlist = []

    strlist = []
    tempstr = ""
    if "[" in string:
        string = string[1:-1]
        #split on highest level ,
        ele_list = []
        level = 0
        tempstr = ""
        for s in string:
            if s == "[":
                level -= 1
            if s == "]":
                level += 1
            if s == "," and level == 0:
                strlist.append(tempstr)
                tempstr = ""
            elif s != " ":
                tempstr += s

    strlist.append(tempstr)
    
    retlist = []
    for string in strlist:
        retlist.append(parse_list(string))

    return retlist

def next_step(steps, num):
    #find smallest step in steps >= num
    for st in steps:
        if st >= num:
            return st
    return num

class defense():
    #a defense is a collection of several things:
    #pads, fakes, and outgoing_steps
    
    #pads: a pad is a distr() that adds incoming packets to a burst
    #a pad is generally not far greater than 0
    #fakes: a fake is a distr() that adds an entirely new burst
    #it is defined by a distr() over incoming packets
    #outgoing_steps: a set of integers that outgoing packets should be padded to
    #fakes just chooses one of those

    #optimize procedures

    def __init__(self):
        self.real_pad = []
        self.real_pad_add = []
        self.fake_pad = []
        self.out_steps = []
        self.out_probs = []
        self.burst_steps = []

    def read_file(self, fname):
        #from a file, read in the defense
        #sometimes some parameters are not indicated by the file
        #so we first load example and overwrite where

        self.load_example()
        func = {"real_pad": [], "real_pad_add": [], "fake_pad": [], "step": [],
                "step_prob": [], "burst_steps": []}

        f = open(fname, "r")
        lines = f.readlines()
        f.close()

        for li in lines:
            if (li != "\n"):
                li = li.split("\t")
                li_type = li[0]
                if (li_type in ["real_pad", "real_pad_add", "fake_pad"]):
                    dType = str(li[1])
                    pmin = float(li[3])
                    pmax = float(li[4])
                    param = parse_list(li[2])
                    d = distr(dType, param, pmin, pmax)
                    func[li_type].append(d)
                if (li_type in ["step", "step_prob", "burst_steps"]):
                    func[li_type] = parse_list(li[1])

        if (func["real_pad"] != []):
            self.real_pad = func["real_pad"]
        if (func["real_pad_add"] != []):
            self.real_pad_add = func["real_pad_add"]
        if (func["fake_pad"] != []):
            self.fake_pad = func["fake_pad"]
        if (func["step"] != []):
            self.out_steps = func["step"]
        if (func["step_prob"] != []):
            self.out_probs = func["step_prob"]
        if (func["burst_steps"] != []):
            self.burst_steps = func["burst_steps"]

    def write_file(self, fname):
        fout = open(fname, "w")
        for r in self.real_pad:
            fout.write("real_pad" + "\t" + repr(r) + "\n")
        for r in self.real_pad_add:
            fout.write("real_pad_add" + "\t" + repr(r) + "\n")
        for r in self.fake_pad:
            fout.write("fake_pad" + "\t" + repr(r) + "\n")

        fout.write("step \t" + repr(self.out_steps) + "\n")
        fout.write("step_prob \t" + repr(self.out_probs) + "\n")
        fout.write("burst_steps \t" + repr(self.burst_steps) + "\n")
        fout.close()

    def load_example(self):

        for i in range(0, 100):
            self.real_pad.append(distr("uniform", [0, 1], 0, 1))
            add_nums = [40, 45, 200]
            probs = [0.2, 0.7, 0.1]
            self.real_pad_add.append(distr("discrete", zip(add_nums, probs), 0, 210))
            self.fake_pad.append(distr("uniform", [0, 100], 0, 100))

        self.out_steps = [50, 200]
        #for fake pad:
        self.out_probs = [0.99, 0.01]
        #pad() outputs this many bursts:
        self.burst_steps = [5, 10, 15, 20, 30, 50]

    def pad(self, sequence):
        #given a sequence of bursts, pad it
        #sequence = [burst1, burst2, ...]
        #burst = [num_outgoing, num_incoming] two integers
        padsequence = []
        #real padding
        for i in range(0, min(len(sequence), len(self.real_pad))):
            burst_out = sequence[i][0]
            burst_inc = sequence[i][1]

            burst_out = next_step(self.out_steps, burst_out)

            burst_inc += int(self.real_pad_add[i].draw()) #int twice here to simplify logic later
            burst_inc = int((1 + self.real_pad[i].draw()) * burst_inc)

            padsequence.append([burst_out, burst_inc])

        #fake bursts
        #we ignore probabilities and pad to a number of bursts
        wanted_burst_num = next_step(self.burst_steps, len(padsequence))
        for i in range(len(padsequence), wanted_burst_num):
            p = random.uniform(0, 1)
            out_i = -1
            while p > 0 and out_i < len(self.out_probs) - 1:
                out_i += 1
                p -= self.out_probs[out_i]
            burst_out = self.out_steps[out_i]
            burst_inc = int(self.fake_pad[i].draw())
            padsequence.append([burst_out, burst_inc])

        return padsequence

    def logprob_pad(self, padsequence, sequence, is_printing = 0):
        #returns log of probability that pad(sequence) generated padsequence
        #if pad is changed, this has to be changed!
        #if it is impossible, return 1 (must capture this case when calling this)

        logp = 0 #this will be negative

        if len(padsequence) < len(sequence):
            return 1

        #real padding:
        for i in range(0, len(sequence)):
            burst_out = sequence[i][0]
            burst_inc = sequence[i][1]

            padburst_out = padsequence[i][0]
            padburst_inc = padsequence[i][1]

            #does outgoing check out?
            if (padburst_out != next_step(self.out_steps, burst_out)):
                return 1
            
            if (padburst_inc < burst_inc):
                return 1
            else:
                p = 0
                if self.real_pad_add[i].dType == "discrete": #save time
                    for add_pair in self.real_pad_add[i].param:
                        add = add_pair[0]
                        p1 = self.real_pad_add[i].prob(add, add+1)
                        if (p1 == 0):
                            p += 0
                        else:
                            p2 = self.real_pad[i].pad_prob(padburst_inc, burst_inc + add)
                            p += p1 * p2
                else:
                    for add in range(self.real_pad_add[i].pmin, self.real_pad_add[i].pmax):
                        p1 = self.real_pad_add[i].prob(add, add+1)
                        if (p1 == 0):
                            p += 0
                        else:
                            p2 = self.real_pad[i].pad_prob(padburst_inc, burst_inc + add)
                            p += p1 * p2
                
                if p <= 0:
                    return 1
                logp += math.log(p)

        real_logp = logp
        iprint("Real padding sum: " + str(real_logp), is_printing)

        #fake padding:
        wanted_burst_num = next_step(self.burst_steps, len(sequence))
        if len(padsequence) != wanted_burst_num:
            return 1

        for i in range(len(sequence), len(padsequence)):
            padburst_out = padsequence[i][0]
            padburst_inc = padsequence[i][1]
            if padburst_out in self.out_steps:
                p = self.out_probs[self.out_steps.index(padburst_out)]
            else:
                return 1
            if p <= 0:
                return 1
            logp += math.log(p)

            p = self.fake_pad[i].prob(padburst_inc, padburst_inc + 1)
            if p <= 0:
                return 1
            logp += math.log(p)
            
        fake_logp = logp - real_logp
        iprint("Fake padding sum: " + str(fake_logp), is_printing)
            
        return logp

    def optimize_multi_perburst(self, dType, real_pad_min, real_pad_max, mpairs_data):
        self.real_pad = []
        
        multi_choices = [] #here there are choices for each burst_i
        for burst_i in range(0, 100):
            multi_choices.append([])        
        
        #gather real_pad numbers:
        CLOSED_SITENUM = len(mpairs_data) - 1 # last is open world
        CLOSED_INSTNUM = len(mpairs_data[0])
        for trial_i in range(0, 500): #more trials because we need more input
            this_i = random.randint(0, CLOSED_SITENUM-1)
            this_j = random.randint(0, CLOSED_INSTNUM-1)
            this_pair = mpairs_data[this_i][this_j]

            that_pairs = []
            for that in range(0, 500):
                that_i = random.randint(0, CLOSED_SITENUM-1)
                that_j = random.randint(0, CLOSED_INSTNUM-1)
                that_pair = mpairs_data[that_i][that_j]
                if len(this_pair) == len(that_pair) and this_i != that_i:
                    that_pairs.append(that_pair)
            if len(that_pairs) < 5:
                continue

            diffs = []
            for that_pair in that_pairs:
                diff = 0
                for pair_i in range(0, len(this_pair)):
                    diff += abs(this_pair[pair_i][1] - that_pair[pair_i][1])
                diffs.append(diff)

            #this is the minimum pair
            that_pair = that_pairs[diffs.index(min(diffs))]

            #now we pick some random additions
            for burst_i in range(0, len(this_pair)):                
                for rand_add_trial in range(0, 10):
                    this_add = self.real_pad_add[burst_i].draw()
                    that_add = self.real_pad_add[burst_i].draw()
                for multi_trial in range(0, 10):
                    large_multi = random.uniform(real_pad_min,
                                                real_pad_min + (real_pad_max - real_pad_min)/4.0)
                    this_inc = this_pair[burst_i][1] + this_add
                    that_inc = that_pair[burst_i][1] + that_add

                    large_inc = max(this_inc, that_inc)
                    small_inc = min(this_inc, that_inc)

                    large_inc = large_inc * (1 + large_multi)
                    small_multi = large_inc/float(small_inc) - 1

                    if (small_multi <= real_pad_max and small_multi >= real_pad_min):
                        multi_choices[burst_i].append(small_multi)

        for burst_i in range(0, 100):
            if len(multi_choices[burst_i]) > 10:
                d = distr()
                d.fit(dType, real_pad_min, real_pad_max, multi_choices[burst_i])
                self.real_pad.append(d)
            else:
                self.real_pad.append(self.real_pad[-1])

            

    def optimize_multi(self, dType, real_pad_min, real_pad_max, mpairs_data):
        #optimizes self.real_pad
        
        real_pad_choices = []
        #gather real_pad numbers:
        CLOSED_SITENUM = len(mpairs_data) - 1 # last is open world
        CLOSED_INSTNUM = len(mpairs_data[0])
        for trial_i in range(0, 100):
            this_i = random.randint(0, CLOSED_SITENUM-1)
            this_j = random.randint(0, CLOSED_INSTNUM-1)
            this_pair = mpairs_data[this_i][this_j]

            that_pairs = []
            for that in range(0, 500):
                that_i = random.randint(0, CLOSED_SITENUM-1)
                that_j = random.randint(0, CLOSED_INSTNUM-1)
                that_pair = mpairs_data[that_i][that_j]
                if len(this_pair) == len(that_pair) and this_i != that_i:
                    that_pairs.append(that_pair)
            if len(that_pairs) < 5:
                continue

            diffs = []
            for that_pair in that_pairs:
                diff = 0
                for pair_i in range(0, len(this_pair)):
                    diff += abs(this_pair[pair_i][1] - that_pair[pair_i][1])
                diffs.append(diff)

            #this is the minimum pair
            that_pair = that_pairs[diffs.index(min(diffs))]

            #now we pick some random additions
            for rand_add_trial in range(0, 30):
                this_add = self.real_pad_add[0].draw()
                that_add = self.real_pad_add[0].draw()

                for multi_trial in range(0, 30):
                    multi = random.uniform(real_pad_min,
                                           real_pad_min + (real_pad_max - real_pad_min)/4.0)
                    temp_multi_choice = []
                    for pair_i in range(0, len(this_pair)):
                        this_inc = this_pair[pair_i][1] + this_add
                        that_inc = that_pair[pair_i][1] + that_add

                        large_inc = max(this_inc, that_inc)
                        small_inc = min(this_inc, that_inc)

                        large_inc = large_inc * (1 + multi)
                        multi_choice = large_inc/float(small_inc) - 1
                        if (multi_choice > real_pad_max or multi_choice < real_pad_min):
                            break

                        temp_multi_choice.append(multi_choice)

                    for choice in temp_multi_choice:
                        real_pad_choices.append(choice)

        d = distr()
        d.fit(dType, real_pad_min, real_pad_max, real_pad_choices)
        self.real_pad = []
        for i in range(0, 100):
            self.real_pad.append(d)

    def optimize_add_perburst(self, ADDNUM, add_min, add_max, wanted_oh, mpairs_data):

        self.real_pad_add = []
        accepted_ratios = [] #accepted_ratio[burst_i]
        
        for burst_i in range(0, 100):
            if (self.real_pad[burst_i].dType == "normal"):
                accepted_ratio = float(self.real_pad[burst_i].param[0] + self.real_pad[burst_i].param[1] + 1)
                #mean + sd covers 70%
            elif (self.real_pad[burst_i].dType == "uniform"):
                a = self.real_pad[burst_i].param[0]
                b = self.real_pad[burst_i].param[1]
                accepted_ratio = (b - a) * 0.7 + a + 1
            elif (self.real_pad[burst_i].dType == "kde"):
                points = sorted(self.real_pad[burst_i].param)
                accepted_ratio = points[int(0.7* len(points))] + 1
            else:
                print "optimize_add_perburst: real_pad type cannot be dealt with"
                sys.exit(0)

            accepted_ratios.append(accepted_ratio)
        
        CLOSED_SITENUM = len(mpairs_data) - 1 #last is open world
        CLOSED_INSTNUM = len(mpairs_data[0])

        for burst_i in range(0, 100):
            add_results = []
            accepted_ratio = accepted_ratios[burst_i]
            for add_choice in range(0, 20):
                #choose ADDNUM random numbers
                add_nums = []
                tried = 0
                for i in range(0, ADDNUM):
                    radd_num = random.randint(add_min, add_max)
                    while (radd_num in add_nums):
                        radd_num = random.randint(add_min, add_max)
                        tried += 1
                    #safeguard
                    if (tried > 1000):
                        print "optimize_add failed: could not pick add_nums"
                        sys.exit(0)
                    add_nums.append(radd_num)
                add_nums = sorted(add_nums)
                
                add_successes = []
                for i in range(0, ADDNUM):
                    add_successes.append([])
                    for j in range(0, ADDNUM):
                        add_successes[-1].append(0)

                has_add = 0
                #populate add_successes
                for add_trial in range(0, 100):
                    #pick two random sequences
                    gotsequence = 0
                    while (gotsequence == 0):
                        this_pair_i = random.randint(0, CLOSED_SITENUM-1)
                        that_pair_i = random.randint(0, CLOSED_SITENUM-1)
                        this_pair_j = random.randint(0, CLOSED_INSTNUM-1)
                        that_pair_j = random.randint(0, CLOSED_INSTNUM-1)
                        if (this_pair_i != this_pair_j):
                            this_pair = mpairs_data[this_pair_i][this_pair_j]
                            that_pair = mpairs_data[that_pair_i][that_pair_j]
                        else:
                            continue
                        if len(this_pair) == len(that_pair) and len(this_pair) != 0:
                            gotsequence = 1

                    #evaluate each possible pair
                    for a_i in range(0, ADDNUM):
                        for a_j in range(0, ADDNUM):
                            if (a_i == a_j):
                                continue

                            if (len(this_pair) > burst_i):
                                this_out = this_pair[burst_i][1] + add_nums[a_i]
                                that_out = that_pair[burst_i][1] + add_nums[a_j]
                                ratio = that_out/float(this_out)
                                if (ratio >= 1/accepted_ratio and ratio <= accepted_ratio):
                                    add_successes[a_i][a_j] += 1
                                    add_successes[a_j][a_i] += 1
                                    has_add = 1

                #now use quadratic programming to find best add_probs
                #notation as on wikipedia (lol)

                if (has_add == 0):
                    continue

                costs = []
                for i in range(0, 20):
                    costs.append(20 * i)

                for cost in costs:

                    Q = - 2 * numpy.matrix(add_successes)
                    E = numpy.ones((1, ADDNUM))
                    ET = numpy.transpose(E)

                    S = numpy.concatenate((Q, ET), axis=1)
                    Sp = numpy.concatenate((E, numpy.zeros((1, 1))), axis=1)
                    S = numpy.concatenate((S, Sp), axis=0)
                    
                    c = - numpy.transpose(numpy.matrix(add_nums) * cost)
                    d = numpy.ones((1, 1))
                    cd = numpy.concatenate((-c, d))

                    try:
                        solution = numpy.linalg.inv(S) * cd
                    except:
                        continue

                    add_probs = []
                    for i in range(0, ADDNUM):
                        add_probs.append(float(solution[i]))

                    for i in range(0, ADDNUM):
                        if (add_probs[i] < 0):
                            add_probs[i] = 0

                    s = sum(add_probs)
                    for i in range(0, ADDNUM):
                        if (s != 0):
                            add_probs[i] /= s
                        else:
                            add_probs[i] = 1/float(ADDNUM)

                    #evaluate effectiveness
                    Suc = numpy.matrix(add_successes)
                    x = numpy.matrix(add_probs)
                    effect = x * Suc * numpy.transpose(x)
                    effect = float(effect)

                    #evaluate overhead

                    oh = numpy.matrix(add_nums) * numpy.transpose(numpy.matrix(add_probs))
                    oh = float(oh)

                    add_results.append([add_nums, add_probs, effect, oh])

            if len(add_results) == 0:
                self.real_pad_add.append(self.real_pad_add[-1])
                continue

            #pareto
            add_results_keep = [1] * len(add_results)
            for i in range(0, len(add_results)):
                for j in range(0, len(add_results)):
                    if (add_results[j][2] > add_results[i][2] and
                        add_results[j][3] <= add_results[i][3]):
                        add_results_keep[i] = 0
                        break
                    
            add_results_kept = []
            for i in range(0, len(add_results)):
                if (add_results_keep[i] == 1):
                    add_results_kept.append(add_results[i])
            add_results = add_results_kept

            add_results = sorted(add_results, key = lambda add_results:add_results[3])

            #return best result below overhead
            wanted_i = len(add_results) - 1
            for add_i in range(0, len(add_results)):
                this_oh = add_results[add_i][3]
                if (this_oh > wanted_oh):
                    wanted_i = add_i
                    break

            if (wanted_i > 0):
                if (wanted_oh - add_results[wanted_i-1][3] <
                    add_results[wanted_i][3] - wanted_oh):
                    wanted_i -= 1
                    
            add_nums = add_results[wanted_i][0]
            add_probs = add_results[wanted_i][1]

            adds = []
            for i in range(0, len(add_probs)):
                if (add_probs[i] != 0):
                    adds.append([add_nums[i], add_probs[i]])

            self.real_pad_add.append(distr("discrete", adds,
                                           add_min, add_max + 10))
                #+10 is to deal with some discrete drawing problem

        

    def optimize_add(self, ADDNUM, add_min, add_max, wanted_oh, mpairs_data):
        #optimizes self.real_pad_add
        
        if (self.real_pad[0].dType == "normal"):
            accepted_ratio = float(self.real_pad[0].param[0] + self.real_pad[0].param[1] + 1)
            #mean + sd covers 70%
        elif (self.real_pad[0].dType == "uniform"):
            a = self.real_pad[0].param[0]
            b = self.real_pad[0].param[1]
            accepted_ratio = (b - a) * 0.7 + a + 1
        elif (self.real_pad[0].dType == "kde"):
            points = sorted(self.real_pad[0].param)
            accepted_ratio = points[int(0.7* len(points))] + 1
        else:
            print "optimize_add: real_pad type cannot be dealt with"
            sys.exit(0)

            
        add_results = []
        CLOSED_SITENUM = len(mpairs_data) - 1 #last is open world
        CLOSED_INSTNUM = len(mpairs_data[0])

        
        for add_choice in range(0, 100):
            #choose ADDNUM random numbers
            add_nums = []
            tried = 0
            for i in range(0, ADDNUM):
                radd_num = random.randint(add_min, add_max)
                while (radd_num in add_nums):
                    radd_num = random.randint(add_min, add_max)
                    tried += 1
                #safeguard
                if (tried > 1000):
                    print "optimize_add failed: could not pick add_nums"
                    sys.exit(0)
                add_nums.append(radd_num)
            add_nums = sorted(add_nums)
            
            add_successes = []
            for i in range(0, ADDNUM):
                add_successes.append([])
                for j in range(0, ADDNUM):
                    add_successes[-1].append(0)
            for add_trial in range(0, 300):
                #pick two random sequences
                gotsequence = 0
                while (gotsequence == 0):
                    this_pair_i = random.randint(0, CLOSED_SITENUM-1)
                    that_pair_i = random.randint(0, CLOSED_SITENUM-1)
                    this_pair_j = random.randint(0, CLOSED_INSTNUM-1)
                    that_pair_j = random.randint(0, CLOSED_INSTNUM-1)
                    if (this_pair_i != this_pair_j):
                        this_pair = mpairs_data[this_pair_i][this_pair_j]
                        that_pair = mpairs_data[that_pair_i][that_pair_j]
                    else:
                        continue
                    if len(this_pair) == len(that_pair) and len(this_pair) != 0:
                        gotsequence = 1

                #evaluate each possible pair
                for a_i in range(0, ADDNUM):
                    for a_j in range(0, ADDNUM):
                        if (a_i == a_j):
                            continue
                        accepted = 1
                        for pair_i in range(0, len(this_pair)):
                            this_out = this_pair[pair_i][1] + add_nums[a_i]
                            that_out = that_pair[pair_i][1] + add_nums[a_j]
                            ratio = that_out/float(this_out)
                            if (ratio >= 1/accepted_ratio and ratio <= accepted_ratio):
                                add_successes[a_i][a_j] += 1
                                add_successes[a_j][a_i] += 1

            #now use quadratic programming to find best add_probs
            #notation as on wikipedia (lol)

            costs = []
            for i in range(0, 20):
                costs.append(20 * i)

            for cost in costs:

                Q = - 2 * numpy.matrix(add_successes)
                E = numpy.ones((1, ADDNUM))
                ET = numpy.transpose(E)

                S = numpy.concatenate((Q, ET), axis=1)
                Sp = numpy.concatenate((E, numpy.zeros((1, 1))), axis=1)
                S = numpy.concatenate((S, Sp), axis=0)
                
                c = - numpy.transpose(numpy.matrix(add_nums) * cost)
                d = numpy.ones((1, 1))
                cd = numpy.concatenate((-c, d))

                try:
                    solution = numpy.linalg.inv(S) * cd
                except:
                    continue

                add_probs = []
                for i in range(0, ADDNUM):
                    add_probs.append(float(solution[i]))

                for i in range(0, ADDNUM):
                    if (add_probs[i] < 0):
                        add_probs[i] = 0

                s = sum(add_probs)
                for i in range(0, ADDNUM):
                    if (s != 0):
                        add_probs[i] /= s
                    else:
                        add_probs[i] = 1/float(ADDNUM)

                #evaluate effectiveness
                Suc = numpy.matrix(add_successes)
                x = numpy.matrix(add_probs)
                effect = x * Suc * numpy.transpose(x)
                effect = float(effect)

                #evaluate overhead

                oh = numpy.matrix(add_nums) * numpy.transpose(numpy.matrix(add_probs))
                oh = float(oh)

                add_results.append([add_nums, add_probs, effect, oh])

        #pareto
        add_results_keep = [1] * len(add_results)
        for i in range(0, len(add_results)):
            for j in range(0, len(add_results)):
                if (i != j):
                    if (add_results[j][2] >= add_results[i][2] and
                        add_results[j][3] <= add_results[i][3]):
                        add_results_keep[i] = 0
                        break
                
        add_results_kept = []
        for i in range(0, len(add_results)):
            if (add_results_keep[i] == 1):
                add_results_kept.append(add_results[i])
        add_results = add_results_kept

        add_results = sorted(add_results, key = lambda add_results:add_results[3])

        #return best result below overhead
        wanted_i = len(add_results) - 1
        for add_i in range(0, len(add_results)):
            this_oh = add_results[add_i][3]
            if (this_oh > wanted_oh):
                wanted_i = add_i
                break

        if (wanted_i > 0):
            if (wanted_oh - add_results[wanted_i-1][3] <
                add_results[wanted_i][3] - wanted_oh):
                wanted_i -= 1
                
        add_nums = add_results[wanted_i][0]
        add_probs = add_results[wanted_i][1]
        self.real_pad_add = []

        adds = []
        for i in range(0, len(add_probs)):
            if (add_probs[i] != 0):
                adds.append([add_nums[i], add_probs[i]])
        for i in range(0, 100):
            self.real_pad_add.append(distr("discrete", adds,
                                           add_min, add_max + 10))
            #+10 is to deal with some discrete drawing problem

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
            print "find_steps_oh: could not find any results"
            sys.exit(0)

        steps_results = sorted(steps_results, key = lambda steps_results: -steps_results[1])
        
        steps = steps_results[0][0]

        return steps
            

    def find_steps(self, STEPNUM, min_collision, nums):
        #nums is a list of numbers
        #return STEPNUM steps that satisfy min_collision with low overhead
        #a step is a number that divides the set
        #a rise is the numbers between two steps

        steps_results = []

        nums = sorted(nums)

        for prob_trial in range(0, 500):
            #get random rise_probs based on the min_collision we need
            this_cover = 0
            possible_min = 0
            possible_max = 1
            remaining_cover = min_collision
            got_probs = 0
            while (got_probs == 0):
                rise_probs = []
                failed = 0
                for i in range(0, STEPNUM - 1):
                    if remaining_cover > 0.5:
                        #simplifies search
                        possible_min = 0.5 + math.sqrt(remaining_cover * 2 - 1) / 2
                    else:
                        possible_min = 0
                    if (possible_min > possible_max):
                        failed = 1
                        break
                    rise_probs.append(random.uniform(possible_min, possible_max))
                    remaining_cover -= rise_probs[-1] * rise_probs[-1]
                    possible_max -= rise_probs[-1]
                if (failed == 0):
                    rise_probs.append(1 - sum(rise_probs))
                    total_collision = 0
                    for o in rise_probs:
                        total_collision += o * o
                    if total_collision >= min_collision:
                        got_probs = 1

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

            stepped_nums = [[]] #stepped_nums[i] is a rise

            current_step_i = 0

            for n_i in range(0, len(nums)):
                if nums[n_i] > steps[current_step_i]:
                    current_step_i += 1
                    stepped_nums.append([])
                stepped_nums[-1].append(nums[n_i])

            oh = 0
                
            for rise in stepped_nums:
                m = max(rise)
                for num in rise:
                    oh += m - num
                    
            steps_results.append([steps, oh])

        steps_results = sorted(steps_results, key = lambda steps_results: steps_results[1])
        steps = steps_results[0][0]

        return steps
            
##    def optimize_out(self, STEPNUM, min_collision, mpairs_data):
    def optimize_out(self, max_oh, mpairs_data):
        #optimizes self.out_steps
        
        #returns the cheapest defense with effectiveness above min_collision
        
        out_nums = []
        CLOSED_SITENUM = len(mpairs_data) - 1
        CLOSED_INSTNUM = len(mpairs_data[0])
        for i in range(0, CLOSED_SITENUM):
            for j in range(0, CLOSED_INSTNUM):
                seq = mpairs_data[i][j]
                for s in seq:
                    out_nums.append(s[0])

        out_nums = sorted(out_nums)

        self.out_steps = self.find_steps_oh(max_oh, out_nums)


    def optimize_fake_perburst(self, dType, fake_min, fake_max, mpairs_data):
        #optimizes self.fake_pad, self.out_probs
        #fake_pad is a distribution of type dType that produces numbers between fake_min and fake_max
        #out_probs is just counting

        #fake_pad:

        old_d = self.fake_pad[0]

        inc_nums = [] #inc_nums[burst_i] will be inc_nums for that burst
        for burst_i in range(0, 100):
            inc_nums.append([])

        CLOSED_SITENUM = len(mpairs_data) - 1
        CLOSED_INSTNUM = len(mpairs_data[0])
        for i in range(0, CLOSED_SITENUM):
            for j in range(0, CLOSED_INSTNUM):
                seq = mpairs_data[i][j]
                for burst_i in range(0, len(seq)):
                    inc_num = seq[burst_i][1]
                    inc_num += int(self.real_pad_add[i].draw())
                    inc_num = int((1 + self.real_pad[i].draw()) * inc_num)
                    if (inc_num < fake_max and inc_num > fake_min):
                        inc_nums[burst_i].append(inc_num)

        self.fake_pad = []
        for burst_i in range(0, 100):
            if (len(inc_nums[burst_i]) > 5):
                d = distr()
                d.fit(dType, fake_min, fake_max, inc_nums[burst_i])
                self.fake_pad.append(d)
                old_d = d
            else:
                self.fake_pad.append(old_d)
        
        out_nums = []
        for i in range(0, CLOSED_SITENUM):
            for j in range(0, CLOSED_INSTNUM):
                seq = mpairs_data[i][j]
                for s in seq:
                    out_nums.append(s[0])
        out_nums = sorted(out_nums)

        out_steps = []
        for o in self.out_steps:
            out_steps.append(o)
        #for safety
        out_steps.append(out_nums[-1])
            
        out_counts = [0] * len(out_steps)

        out_steps_i = 0

        for i in range(0, len(out_nums)):
            while (out_nums[i] > out_steps[out_steps_i]):
                out_steps_i += 1
            out_counts[out_steps_i] += 1

        #kick off the last safety
        out_counts = out_counts[:-1]

        self.out_probs = []
        s = float(sum(out_counts))
        for c in out_counts:
            self.out_probs.append(c/s)
            
    def optimize_fake(self, dType, fake_min, fake_max, mpairs_data):
        #optimizes self.fake_pad, self.out_probs
        #fake_pad is a distribution of type dType that produces numbers between fake_min and fake_max
        #out_probs is just counting

        #fake_pad:

        inc_nums = []

        CLOSED_SITENUM = len(mpairs_data) - 1
        CLOSED_INSTNUM = len(mpairs_data[0])
        for i in range(0, CLOSED_SITENUM):
            for j in range(0, CLOSED_INSTNUM):
                seq = mpairs_data[i][j]
                for s in seq:
                    inc_num = s[1]
                    inc_num += int(self.real_pad_add[i].draw())
                    inc_num = int((1 + self.real_pad[i].draw()) * inc_num)
                    if (inc_num < fake_max and inc_num > fake_min):
                        inc_nums.append(inc_num)

        if (len(inc_nums) > 5):
            d = distr()
            d.fit(dType, fake_min, fake_max, inc_nums)
            self.fake_pad = []
            for i in range(0, 100):
                self.fake_pad.append(d)

        #else optimization has failed and we simply don't change fake_pad
        #this is because the parameters are in conflict with real_pad/real_pad_add
        
        out_nums = []
        for i in range(0, CLOSED_SITENUM):
            for j in range(0, CLOSED_INSTNUM):
                seq = mpairs_data[i][j]
                for s in seq:
                    out_nums.append(s[0])
        out_nums = sorted(out_nums)

        out_steps = []
        for o in self.out_steps:
            out_steps.append(o)
        #for safety
        out_steps.append(out_nums[-1])
            
        out_counts = [0] * len(out_steps)

        out_steps_i = 0

        for i in range(0, len(out_nums)):
            while (out_nums[i] > out_steps[out_steps_i]):
                out_steps_i += 1
            out_counts[out_steps_i] += 1

        #kick off the last safety
        out_counts = out_counts[:-1]

        self.out_probs = []
        s = float(sum(out_counts))
        for c in out_counts:
            self.out_probs.append(c/s)
        

##    def optimize_burst(self, BURSTNUM, min_collision, mpairs_data):
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

class detdefense():
    #much like defense()
    #inherit from defense?

    def __init__(self):
        #note that out_steps here and in defense() are different formats
        self.inc_steps = []
        self.inc_probs = []
        self.out_steps = []
        self.out_probs = []

        #we actually don't use probs yet

    def read_file(self, fname):
        #from a file, read in the defense
        #sometimes some parameters are not indicated by the file
        #so we first load example and overwrite where

        self.load_example()
        func = {"inc_steps": [], "inc_probs": [],
                "out_steps": [], "out_probs": []}

        f = open(fname, "r")
        lines = f.readlines()
        f.close()

        for li in lines:
            if (li != "\n"):
                li = li.split("\t")
                li_type = li[0]
                func[li_type] = parse_list(li[1])
        
        if (func["inc_steps"] != []):
            self.inc_steps = func["inc_steps"]
        
        if (func["inc_probs"] != []):
            self.inc_probs = func["inc_probs"]
        
        if (func["out_steps"] != []):
            self.out_steps = func["out_steps"]
        
        if (func["out_probs"] != []):
            self.out_probs = func["out_probs"]

    def write_file(self, fname):
        fout = open(fname, "w")
        for r in self.inc_steps:
            fout.write("inc_steps" + "\t" + repr(r) + "\n")
        for r in self.inc_probs:
            fout.write("inc_probs" + "\t" + repr(r) + "\n")
        for r in self.out_steps:
            fout.write("out_steps" + "\t" + repr(r) + "\n")
        for r in self.inc_steps:
            fout.write("inc_steps" + "\t" + repr(r) + "\n")            
        fout.close()

    def load_example(self):
        self.out_steps = []
        self.out_probs = []
        self.inc_steps = []
        self.inc_probs = []
        for i in range(0, 100):
            self.out_steps.append([50, 200])
            self.out_probs.append([0.99, 0.01])
            self.inc_steps.append([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
            self.inc_probs.append([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    def pad(self, sequence):
        #performs deterministic padding
        padsequence = []

        #real padding
        for i in range(0, len(sequence)):
            burst_out = next_step(self.out_steps[i], sequence[i][0])
            burst_inc = next_step(self.inc_steps[i], sequence[i][1])
            padsequence.append([burst_out, burst_inc])

        #fake padding

##        burst_num = len(padsequence)
##        wanted_burst_num = self.next_step(self.burst_steps, burst_num)
##        for i in range(burst_num, wanted_burst_num):
##            p = random.uniform(0, 1)
##            inc_i = -1
##            while p > 0 and inc_i < len(self.inc_probs) - 1:
##                inc_i += 1
##                p -= self.inc_probs[inc_i]
##            burst_inc = self.inc_steps[inc_i]
##            
##            p = random.uniform(0, 1)
##            out_i = -1
##            while p > 0 and out_i < len(self.out_probs) - 1:
##                out_i += 1
##                p -= self.out_probs[out_i]
##            burst_out = self.out_steps[out_i]
##            padsequence.append([burst_out, burst_inc])

        return padsequence

    def optimize_det_perburst(self, is_inc, stepnums, mpairs_data):
        #stepnums is a list of stepnum to use for each burst

        if len(stepnums) < 100:
            print "optimize_det_perburst: stepnums is too short"
            sys.exit(0)
        
        if (is_inc != 0):
            is_inc = 1

        #define defaults in case not enough nums
        if (is_inc == 1):
            if len(self.inc_steps) > 0:
                default_steps = self.inc_steps[0]
                default_probs = self.inc_probs[0]
            #else just pray
            self.inc_steps = []
            self.inc_probs = []
            steps = self.inc_steps
            probs = self.inc_probs
        if (is_inc == 0):
            if len(self.out_steps) > 0:
                default_steps = self.out_steps[0]
                default_probs = self.out_probs[0]
            self.out_steps = []
            self.out_probs = []
            steps = self.out_steps
            probs = self.out_probs

            #default still has it

        nums = [] #nums[i] are the nums for burst i
        for i in range(0, 100):
            nums.append([])
        for site in mpairs_data:
            for sinste in site:
                for burst_i in range(0, len(sinste)):
                    nums[burst_i].append(sinste[burst_i][is_inc])

        

        for burst_i in range(0, 100):
            if len(nums[burst_i]) > 10:
                #optimize properly

                nums_max = max(nums[burst_i])

                #limit to 150
                if len(nums[burst_i]) > 150:
                    trainnums = random.sample(nums[burst_i], 150)
                else:
                    trainnums = nums[burst_i]
                trainnums = sorted(trainnums)

                #get steps
                this_steps = []
                step_inds, oh = split(trainnums, stepnums[burst_i]) #from split.py
                for step_ind in step_inds:
                    this_steps.append(trainnums[step_ind-1])
                if this_steps[-1] != nums_max:
                    this_steps.append(nums_max)

                #get size of each step in counts
                counts = []
                this_count = 0
                steps_pointer = 0
                for n in nums[burst_i]:
                    while (this_steps[steps_pointer] < n):
                        steps_pointer += 1
                        counts.append(this_count)
                        this_count = 0
                    this_count += 1

                counts.append(this_count)
                while (len(counts) < len(steps)):
                    counts.append(0)

                #counts -> probs
                s = sum(counts)
                this_probs = []
                for c in counts:
                    this_probs.append(c/float(s))
                
                steps.append(this_steps)
                probs.append(this_probs)
                default_steps = this_steps
                default_probs = this_probs
            else:
                #just use an old one
                steps.append(default_steps)
                probs.append(default_probs)
                
        

    def optimize_det(self, is_inc, stepnum, mpairs_data):
        if (is_inc != 0):
            is_inc = 1
        #optimize inc and out probs and steps

        nums = []
        for site in mpairs_data:
            for sinste in site:
                for burst in sinste:
                    nums.append(burst[is_inc])

        nums = sorted(nums)

        nums_max = max(nums)

        #limit to 300
        if len(nums) > 300:
            trainnums = random.sample(nums, 300)
        else:
            trainnums = nums
        trainnums = sorted(trainnums)

        #get steps
        step_inds, oh = split(trainnums, stepnum) #from split.py

##        print trainnums, stepnum
##
##        print oh/float(len(trainnums))
##        print sum(trainnums)/float(len(trainnums))
        #convert step_inds to steps
        steps = []
        for step_ind in step_inds:
            steps.append(trainnums[step_ind-1]) #urgh
        if steps[-1] != nums_max:
            steps.append(nums_max)

        #get size of each step in counts
        counts = []
        this_count = 0
        steps_pointer = 0
        for n in nums:
            while (steps[steps_pointer] < n):
                steps_pointer += 1
                counts.append(this_count)
                this_count = 0
            this_count += 1

        counts.append(this_count)
        while (len(counts) < len(steps)):
            counts.append(0)

        #counts -> probs
        s = sum(counts)
        probs = []
        for c in counts:
            probs.append(c/float(s))

        #now assign
        if is_inc == 1:
            self.inc_probs = []
            self.inc_steps = []
            for i in range(0, 100):
                self.inc_probs.append(probs)
                self.inc_steps.append(steps)
        if is_inc == 0:
            self.out_probs = []
            self.out_steps = []
            for i in range(0, 100):
                self.out_probs.append(probs)
                self.out_steps.append(steps) 
        

##d = defense()
##d.load_example()
##d.write_file("defense_example")
##d.read_file("defense_normal")
##
##sequence = [[2, 2], [4, 18], [5, 109], [24, 556], [1, 740], [55, 3193], [12, 967], [1, 48],
##            [1, 36], [1, 198], [2, 687], [26, 1285], [2, 813], [1, 4], [4, 11], [2, 10], [1, 1],
##            [1, 7], [4, 2], [1, 4]]
##padsequence = d.pad(sequence)
##
##print d.logprob_pad(padsequence, sequence)
##sequence = [[17, 100], [44, 353], [3, 21], [1, 2]]
##print sequence
##padsequence = d.pad(sequence)
##print padsequence
##logprob = d.logprob_pad(padsequence, sequence)
##print logprob
