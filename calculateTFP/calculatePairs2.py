import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd

def readFile(filename):
    fileContent = pd.read_csv(filename, sep = '\t')
    return fileContent

def getNewAppoint(appoint):
    columns = ['clue_id', 'create_time', 'user_id']
    tempAppoint = appoint[columns].dropna()
    return tempAppoint

def getCluePairs(appoint):
    print 'grouping...'
    groups = appoint.sort_values('create_time').groupby('user_id', sort = False)
    pairs = []
    print 'shifting...'
    count = 0
    for idx, g in groups:
        print str(count) + ' / 580468'
        count = count + 1
        clues = list(g['clue_id'])
        for i in range(len(clues) - 1):
            j = i + 1
            pairs.append([clues[i], clues[j]]) 
    print 'concating...'
    pairs = pd.DataFrame(pairs, index=False, columns=['clue_id','clue_id_2'])
    pairs.to_csv('pairs.tsv')
    print 'done'

if __name__=="__main__":
      appointFile = 'hl_appoint.tsv'
      appoint = readFile(appointFile)
      ap = getNewAppoint(appoint)
      getCluePairs(ap)
