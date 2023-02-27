def splittwo(sizes):
    #this is not necessary, does it save time?
    #returns the index of where the split should be when there are only 2 cases
    #Index: sizes list (original input)
    #Range of outputs: 0 to len(sizes) (len(sizes)+1 possibilities)
    #assumes sizes is sorted
##    print "splittwo called", sizes

    
    cost = 0
    costs = []
    for i in range(0, len(sizes)):
        cost += sizes[-1] - sizes[i]
    costs.append(cost)
    for i in range(0, len(sizes)):
        cost -= sizes[-1] - sizes[i]
        if (i != 0):
            cost += i * (sizes[i] - sizes[i-1])
        costs.append(cost)
        
    return [costs.index(min(costs)), len(sizes)], min(costs)

def splitone(sizes):
##    print "splitone called", sizes
    cost = len(sizes) * sizes[-1] - sum(sizes)
    return [len(sizes)], cost

#The central argument for this recurrence strategy is:
#If [a, b, c, d] are optimal for sizes with splitnum = 4
#Then [b, c, d] are optimal for sizes-(things covered by a) with splitnum = 3
#(Proof: Otherwise, [a, b', c', d'] would be better than [a, b, c, d])

def split(sizes, splitnum):
    s = sorted(sizes)
    return split_recur(s, splitnum)

def split_recur(sizes, splitnum):
    #we don't want to sort every time so:
    if (sizes != sorted(sizes)):
        print "split: sizes must be sorted"
        sys.exit(0)

    #first, if all elements are the same, we can return immediately
    has_different = 0
    for i in range(1, len(sizes)):
        if sizes[i] != sizes[i-1]:
            has_different = 1
            break

    if (has_different == 0):
        return [len(sizes)], 0

    #finds optimal splits (and cost) of sizes for splitnum splits
    #uses naive tree-splitting
    #should be n^logk
    if len(sizes) == 2 and splitnum > 2:
        splitnum = 2
    if len(sizes) == 1 and splitnum > 1:
        splitnum = 1

    #base cases
    if (splitnum == 1):
        return splitone(sizes)
    elif (splitnum == 2):
        return splittwo(sizes) #??
    #divide sizes into all possibilities of two roughly equal split() calls
    else:
        splitnum_one = splitnum/2
        splitnum_two = splitnum-splitnum_one
        sizes_one = []
        for i in range(0, len(sizes)):
            sizes_one.append(sizes[i])
        sizes_two = []
        costs = [] #cost of each possible split
        inds = [] #locations of splits

        #cycle over all possible splits
        for i in range(1, len(sizes)):
            #move last element from sizes_one to sizes_two
            sizes_two = sizes_two[::-1]
            sizes_two.append(sizes_one.pop(-1))
            sizes_two = sizes_two[::-1]

            if (sizes[i] != sizes[i-1]):
                #this will happen at least once because we checked above
                ind_one, cost_one = split_recur(sizes_one, splitnum_one)
                ind_two, cost_two = split_recur(sizes_two, splitnum_two)

                inds.append([])
                costs.append(cost_one + cost_two)
                for ind in ind_one:
                    inds[-1].append(ind)
                #inds[-1].append(len(sizes_one))
                for ind in ind_two:
                    inds[-1].append(ind+len(sizes_one))

        min_ind = costs.index(min(costs))
##        print costs, inds
        return inds[min_ind], costs[min_ind]


##import time
##import random
##
##sizes = []
##for n in range(114, 500):
##    for k in range(7, 8):
##        sizes = []
##        for size_i in range(0, n):
##            sizes.append(random.randint(0, n))
##        sizes = sorted(sizes)
##        t1 = time.time()
##        inds, cost = split(sizes, k)
##        print inds, cost
##
##        #challenge optimal cost: generate some random splits
##        inds = []
##        for i in range(0, 7):
##            inds.append(random.randint(0, n))
##        inds = sorted(inds)
##        inds.append(sizes[-1])
##
##        i_pointer = 0
##
##        cost = 0
##
##        for s in sizes:
##            while (s > inds[i_pointer]):
##                i_pointer += 1
##            cost += inds[i_pointer] - s
##        print inds, cost
##            
##        
##        t2 = time.time()
##        print n, k, (t2-t1)/5


##a = [1, 2, 3, 3, 4, 5, 6, 7, 8, 8, 9, 12, 15, 15]
##print split(a, 3)
