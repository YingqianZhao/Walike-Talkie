def cclass_to_acc(cclass):
    #cclass is a list of collision classes
    #from the collision classes, get the attacker accuracy
    accs = [] #one for each cclass
    for c in cclass:
        #c is a list of classifications

        ccount = []
        clabel = []
        for c_i in c:
            if c_i in clabel:
                ccount[clabel.index(c_i)] += 1
            else:
                ccount.append(1)
                clabel.append(c_i)

        for i in range(0, max(ccount)):
            accs.append(1)

        for i in range(max(ccount), sum(ccount)):
            accs.append(0)

    return sum(accs)/float(len(accs))
