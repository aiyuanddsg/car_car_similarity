import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
from datetime import datetime

def readFile(filename):
    fileContent = pd.read_csv(filename, sep = '\t')
    return fileContent

def getAppointNoDealer(appoint):
    appointNoDealer = appoint[appoint['dealer'] == 0]
    return appointNoDealer

def getAppointDealer(appoint):
    appointDealer = appoint[appoint['dealer'] == 3]
    return appointDealer

def getNewAppoint(appoint, feature):
    columns = ['clue_id', 'create_time', 'user_id']
    tempAppoint = appoint[columns].dropna()
    return tempAppoint

def getNewCar(car, feature):
    columns = ['clue_id']
    columns.extend(feature)
    newCar = car[columns]
    newCar = newCar[newCar[feature] > 0]
    return newCar 

def getCluePairs(appoint):
    groups = appoint.sort_values('create_time').groupby('user_id', sort = False)
    pairs = []
    for idx, g in groups:
	clues = list(g['clue_id'])
        for i in range(len(clues) - 1):
            j = i + 1
            pairs.append([clues[i], clues[j]])
    pairs = pd.DataFrame(pairs, columns=['clue_id','clue_id_2'])
    return pairs
    
def addNewFeatureToCar(car, feature):
    now = datetime.now()
    newFeature = feature[1] + '_time_length'
    car[newFeature] = (now.year - car[feature[0]]) * 12 + now.month - car[feature[1]] 
    return newFeature, car
                            
def calculateTransformationProbability(pairs, car, feature):
    pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id').dropna()
    pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2').dropna()
    feature_2 = feature + '_2'
    transformationProbability = pd.crosstab(pairs[feature], pairs[feature_2], normalize=True).round(7) * 1000.0
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
        print feature
        newAppoint = getNewAppoint(appoint, feature)   
        pairs = getCluePairs(newAppoint)
        nCar = getNewCar(car, feature)
        newFeature, newCar = addNewFeatureToCar(nCar, feature)
        transformationProbability = calculateTransformationProbability(pairs, newCar, newFeature)
        fileName = getFileName(newFeature, isDealer)
        writeToFile(fileName, transformationProbability)

if __name__=="__main__":
      catgoryFeatures = [['license_date', 'license_month'], ['business_insurance_year', 'business_insurance_month'], ['strong_insurance_year', 'strong_insurance_month'], ['audit_year', 'audit_month']]
      carFile = 'hl_car.tsv'
      appointFile = 'hl_appoint.tsv'
      car = readFile(carFile)
      appoint = readFile(appointFile)
      appointNoDealer = getAppointNoDealer(appoint)
      appointDealer = getAppointDealer(appoint)
      calculateAllTransformationProbability(appointNoDealer, car, catgoryFeatures, False)
      calculateAllTransformationProbability(appointDealer, car, catgoryFeatures, True)
      
