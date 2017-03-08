import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd

def readFile(filename):
    fileContent = pd.read_csv(filename, sep = '\t')
    return fileContent

def getAppointNoDealer(appointND):
    appointND = appointND[appointND['dealer'] == 0]
    return appointND

def getAppointNotSureDealer(appointND):
    appointND = appointND[appointND['dealer'] != 0]
    return appointND

def getAppointDealer(appointD):
    appointD = appointD[appointD['dealer'] == 3]
    return appointD

def getNewAppoint(appointOld):
    columns = ['clue_id', 'create_time', 'user_id']
    tempAppoint = appointOld[columns].dropna()
    return tempAppoint

def getNewCar(feature1, car1):
    newCar1 = car1[['clue_id', feature1]]
    newCar1 = newCar1[newCar1[feature1] > 0]
    return newCar1

def getCluePairs(appointC):
    groups = appointC.sort_values('create_time').groupby('user_id', sort = False)
    pairs = []
    for idx, g in groups:
	clues = list(g['clue_id'])
        for i in range(len(clues) - 1):
            j = i + 1
            pairs.append([clues[i], clues[j]])
    pairs = pd.DataFrame(pairs, columns=['clue_id','clue_id_2'])
    return pairs

def calculateTransformationProbability(pairs2, car2, feature2):
    pairs2 = pairs2.join(car2.set_index('clue_id')[feature2], on = 'clue_id').dropna()
    pairs2 = pairs2.join(car2.set_index('clue_id')[feature2], on = 'clue_id_2', rsuffix = '_2').dropna()
    feature_2 = feature2 + '_2'
    transformationProbability2 = pd.crosstab(pairs2[feature2], pairs2[feature_2])
    for i in range(len(transformationProbability2)):
        transformationProbability2.iloc[i] = transformationProbability2.iloc[i] / transformationProbability2.iloc[i].max()
    transformationProbability2 = transformationProbability2.round(3)
    return transformationProbability2

def getFileName(feature3, isDealer1):
    if isDealer1 == True:
        return '../../data/categoryTransFormationProbability/' + feature3 + 'Dealer.tsv'
    else:
        return '../../data/categoryTransFormationProbability/' + feature3 + 'NoDealer.tsv'

def getFileName1(feature3):
    return '../../data/categoryTransFormationProbability/' + feature3 + 'NotSureDealer.tsv'

def writeToFile(filename1, transformationProbability1):
    transformationProbability1.to_csv(filename1, sep = '\t')

def calculateAllTransformationProbability(appoint4, car4, catgoryFeatures4, isDealer4):
    for feature4 in catgoryFeatures4:
        print feature4
        newAppoint4 = getNewAppoint(appoint4)
        print 'pairing......'
        print len(appoint4)
        print len(newAppoint4)
        pairs4 = getCluePairs(newAppoint4)
        newCar4 = getNewCar(feature4, car4)
        print 'calculate transformation Probability...'
        transformationProbability4 = calculateTransformationProbability(pairs4, newCar4, feature4)
        fileName4 = getFileName1(feature4)
        writeToFile(fileName4, transformationProbability4)
        print 'done!'

if __name__=="__main__":
      catgoryFeatures = ['emission_standard', 'car_year', 'transfer_num', 'minor_category_id', 'source_level', 'auto_type', 'carriages', 'fuel_type', 'gearbox', 'car_color', 'guobie', 'seats']
      carFile = '../../data/hl_car.tsv'
      appointFile = '../../data/hl_appoint.tsv'
      car = readFile(carFile)
      appoint = readFile(appointFile)
      print len(appoint)
      appointNoDealer = getAppointNoDealer(appoint)
      print len(appointNoDealer)
      appointDealer = getAppointDealer(appoint)
      appointNotSureDealer = getAppointNotSureDealer(appoint)
      print len(appointNotSureDealer)
      #calculateAllTransformationProbability(appointNoDealer, car,  catgoryFeatures, False)
      #calculateAllTransformationProbability(appointDealer, car, catgoryFeatures, True)
      calculateAllTransformationProbability(appointNotSureDealer, car, catgoryFeatures, True)
      
