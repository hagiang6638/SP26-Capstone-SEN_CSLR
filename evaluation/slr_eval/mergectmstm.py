#!/usr/bin/env python

import sys
import pdb

ctmFile = sys.argv[1]
stmFile = sys.argv[2]

ctm = open(ctmFile, "r")
stm = open(stmFile, "r")

ctmDict = []
stmDict = []
addedlines = 0

for line in ctm:
    l = line.strip().split()
    if not l: continue
    ctmDict.append(l)

idx = 0
for line in stm:
    l = line.strip().split()
    if not l: continue
    stmDict.append(l)
    
    # check that ctm line is also valid
    if len(ctmDict) > idx + addedlines and len(ctmDict[idx + addedlines]) > 0 and ctmDict[idx + addedlines][0] == l[0]:  # ctm and stm match:
        if len(ctmDict) > idx + addedlines + 1:
            while (len(ctmDict) > idx + addedlines + 1) and len(ctmDict[idx + addedlines + 1]) > 0 and (ctmDict[idx + addedlines + 1][0] == l[0]):
                addedlines += 1
    else:
        ctmDict.insert(idx + addedlines, [l[0], "1 0.000 0.030 [EMPTY]"])
    idx += 1

stm.close()
ctm.close()
ctm = open(ctmFile, "w+")

for l in ctmDict:
    ctm.write(" ".join(l) + "\n")
ctm.close()
