#.htor is the default format

#File types:
#(from more info to less info)
#.tor: Includes headers and all packet cells. Should be deleted. Starting format.
#.htor: Includes only headers and burst starts. Stored.
#.cell: Standard cell format for attackers. Does NOT have burst starts.
#.burst: Short format for our analysis. Has burst starts.

#.tor -> .htor -> .cell or .burst
#those are one-directional, defined by format1_to_format2 functions

#Data types in memory:
#
#mcells: cell[i] is the ith cell [time, direction]. used for attacks
#mbursts: burst[i] is a list of [directions] of that burst. used for defense

#mcells <-> cells, mbursts <-> bursts
#those are two-directional, defined by seq = read(f) and write(f, seq) functions


def tor_to_htor(fname):
    if fname[-4:] != ".tor":
        print "tor_to_htor format incorrect"
        sys.exit(0)

    tname = fname[:-4] + ".htor"
    fin = open(fname, "r")
    fout = open(tname, "w")
    for li in fin.readlines():
        if len(li) > 0:
            if li[0] != "\t":
                fout.write(li)
    fin.close()
    fout.close()

def htor_to_mcells(fname):
    mcells = []
    if fname[-5:] != ".htor":
        print "htor_to_cell format incorrect"
        sys.exit(0)

    f = open(fname, "r")
    lines = f.readlines()
    f.close()

    for li in lines:
        psize = 0
        if "INCOMING" in li:
            psize = -1
        if "OUTGOING" in li:
            psize = 1
        if psize != 0:
            time = float(li.split(" ")[0])
            mcells.append([time, psize])

    return mcells

def read_mcells(fname):
    if fname[-5:] != ".cell":
        print "read_mcells format incorrect"
        sys.exit(0)

    tname = fname[:-5] + ".cell"
    fin = open(fname, "r")
    lines = fin.readlines()
    fin.close()

    mcells = []

    for li in lines:
        li = li.split("\t")
        mcells.append([float(li[0]), int(li[1])])

    return mcells
    

def write_mcells(fname, mcells):
    if fname[-5:] != ".cell":
        print "write_mcells format incorrect"
        sys.exit(0)

    fout = open(fname, "w")
    for c in mcells:
        fout.write(repr(c[0]) + "\t" + str(c[1]) + "\n")
    fout.close()

def htor_to_mbursts(fname):
    if fname[-5:] != ".htor":
        print "htor_to_mbursts format incorrect"
        sys.exit(0)

    f = open(fname, "r")
    lines = f.readlines()
    f.close()

    #for mbursts we ignore sendmes and ends

    mbursts = [[]]
    for li in lines:
        if "Start" in li:
            if -1 in mbursts[-1]:
                mbursts.append([])
        if "DATA" in li or "BEGIN" in li:
            if "INCOMING" in li:
                mbursts[-1].append(-1)
            if "OUTGOING" in li:
                mbursts[-1].append(1)

    return mbursts

def read_mbursts(fname):
    if fname[-6:] != ".burst":
        print "write_mbursts format incorrect"
        sys.exit(0)

    f = open(fname, "r")
    lines = f.readlines()
    f.close()

    mbursts = []

    for li in lines[:-1]:
        mbursts.append([])
        li = li.split(",")[:-1] #last one is "\n"
        for p in li:
            mbursts[-1].append(int(p))

    return mbursts

def write_mbursts(fname, mbursts):
    if fname[-6:] != ".burst":
        print "write_mbursts format incorrect"
        sys.exit(0)

    fout = open(fname, "w")
    for burst in mbursts:
        for p in burst:
            fout.write(str(p) + ",")
        fout.write("\n")
    fout.close()

def mbursts_to_mpairs(mbursts):
    #mpair just coalesces mburst
    mpairs = []
    for burst in mbursts:
        out_count = burst.count(1)
        in_count = burst.count(-1)
        mpairs.append([out_count, in_count])
    return mpairs

def mcells_to_msizes(mcells):
    #msizes is just mcells without the time, used for clLev
    msizes = []
    for cell in mcells:
        msizes.append(cell[1])

    return msizes

def write_msizes(fname, msizes):
    if fname[-5:] != ".size":
        print "write_msizes format incorrect"
        sys.exit(0)
    fout = open(fname, "w")
    for size in msizes:
        fout.write(str(size) + "\n")
    fout.close()

def read_msizes(fname):
    if fname[-5:] != ".size":
        print "read_msizes format incorrect"
        sys.exit(0)

    fin = open(fname, "r")
    lines = fin.readlines()
    fin.close()

    msizes = []
    for li in lines:
        msizes.append(int(li))
    return msizes


