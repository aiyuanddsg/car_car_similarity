import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd

def readFile(filename):
    fileContent = pd.read_csv(filename, sep = '\t')
    return fileContent

def getAppointNoDealer(appoint):
    appointNoDealer = appoint[appoint['dealer'] == 0]
    return appointNoDealer

def getAppointDealer(appoint):
    appointDealer = appoint[appoint['dealer'] == 3]
    return appointDealer

def getNewAppoint(appoint):
    columns = ['clue_id', 'create_time', 'user_id']
    tempAppoint = appoint[columns].dropna()
    return tempAppoint

def getNewCar(car, feature):
    car = car[car[feature] > 0]
    return car

def getCluePairs(appoint):
    groups = appointC.sort_values('create_time').groupby('user_id', sort = False)
    pairs = []
    for idx, g in groups:
	clues = list(g['clue_id'])
        for i in range(len(clues) - 1):
            j = i + 1
            pairs.append([clues[i], clues[j]])
    pairs = pd.DataFrame(pairs, columns=['clue_id','clue_id_2'])
    return pairs

def setBins():
    bins = pd.DataFrame({'evaluate_score': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'air_displacement': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 'road_haul': [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 'suggest_price': [0, 50000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 10000000], 'base_price': [0, 50000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 10000000], 'price': [0, 50000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 10000000]})
    return bins

def cutFeature(bins, car, feature):
    nCar = car[['clue_id',feature]]
    newCar = pd.DataFrame(pd.cut(nCar[feature], bins[feature]))
    ft = feature + '_a'
    newCar.rename(columns={feature: ft}, inplace = True)
    newCar = pd.concat([nCar, newCar], axis=1)
    return ft, newCar

def calculateTransformationProbability(pairs, car, feature):
    pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id').dropna()
    pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2').dropna()
    feature_2 = feature + '_2'
    transformationProbability = pd.crosstab(pairs[feature], pairs[feature_2], normalize=True).round(3)
    return transformationProbability

def getFileName(feature, isDealer):
    if isDealer == True:
        return 'categoryTransFormationProbability/' + feature + 'Dealer.tsv'
    else:
        return 'categoryTransFormationProbability/' + feature + 'NoDealer.tsv'

def writeToFile(filename, transformationProbability):
    transformationProbability.to_csv(filename, sep = '\t')

def calculateAllTransformationProbability(appoint, car, catgoryFeatures, isDealer):
    for feature in catgoryFeatures:
          newAppoint = getNewAppoint(appoint)
          nCar = getNewCar(car, feature)
          pairs = getCluePairs(newAppoint)
          bins = setBins()
          ft, newCar = cutFeature(bins, nCar, feature)
          transformationProbability = calculateTransformationProbability(pairs, newCar, ft)
          fileName = getFileName(ft, isDealer)
          writeToFile(fileName, transformationProbability)

if __name__=="__main__":
      catgoryFeatures = ['evaluate_score', 'air_displacement', 'road_haul', 'suggest_price', 'base_price', 'price']
      carFile = 'hl_car.tsv'
      appointFile = 'hl_appoint.tsv'
      car = readFile(carFile)
      appoint = readFile(appointFile)
      appointNoDealer = getAppointNoDealer(appoint)
      appointDealer = getAppointDealer(appoint)
      calculateAllTransformationProbability(appointNoDealer, car, catgoryFeatures, False)
      calculateAllTransformationProbability(appointDealer, car, catgoryFeatures, True)
      
