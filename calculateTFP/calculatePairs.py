import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd

def readFile(filename):
    fileContent = pd.read_csv(filename, sep = '\t')
    return fileContent

def getNewAppoint(appoint):
    columns = ['clue_id', 'create_time', 'user_id', 'dealer']
    tempAppoint = appoint[columns].dropna()
    return tempAppoint

def getDealerAppoint(appoint):
    appoint = appoint[appoint['dealer'] == 3]
    return appoint

def getNoDealerAppoint(appoint):
    appoint = appoint[appoint['dealer'] == 0]
    return appoint

def getCluePairs(appoint):
    groups = appoint.sort_values('create_time').groupby('user_id', sort = False)
    pairs = []
    for idx, g in groups:
	clues = list(g['clue_id'])
        for i in range(len(clues) - 1):
            j = i + 1
            pairs.append([clues[i], clues[j]])
    return pairs

def count_same_pairs(pairsD, pairsND):
    count = 0;
    for p in pairsD:
        if p in pairsND:
            count = count + 1
    return count
if __name__=="__main__":
      appointFile = 'hl_appoint.tsv'
      appoint = readFile(appointFile)
      ap = getNewAppoint(appoint)
      apD = getDealerAppoint(ap)
      apND = getNoDealerAppoint(ap)
      pairsD = getCluePairs(apD)
      pairsND = getCluePairs(apND)
      print count_same_pairs
