import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
import random as rd

def readFile(filename):
    fileContent = pd.read_csv(filename, sep = '\t')
    return fileContent

def getAppointNoDealer(appointND):
    appointND = appointND[appointND['dealer'] == 0]
    return appointND

def getNewAppoint(appointOld):
    columns = ['clue_id', 'create_time', 'user_id']
    tempAppoint = appointOld[columns].dropna()
    return tempAppoint

def getNewCar(feature1, car1):
    newCar1 = car1[['clue_id', feature1]]
    newCar1 = newCar1[newCar1[feature1] > 0]
    return newCar1

def get_user_pairs(appoint, car):
    ap = appoint.join(car.set_index('clue_id')['price'], on = 'clue_id')
    ap = ap[['user_id', 'price']]
    mean_price = ap.groupby('user_id').mean()
    mean_price = mean_price[mean_price['price'] > 0]
    amp = mean_price.sort_values('price', ascending=True)
    idx = amp.index
    user_pairs = []
    count = 0
    for i in range(0, 5000000):
        j = rd.randint(0, len(idx) / 2)
        k = rd.randint(len(idx) / 2 + 1, len(idx) - 1)
        if amp.ix[idx[j]]['price'] * 4 < amp.ix[idx[k]]['price']:
            if rd.random() < 0.5:
                user_pairs.append([idx[j], idx[k]])
            else:
                user_pairs.append([idx[k], idx[j]])
    return user_pairs

def get_clue_pairs(user_pairs, appoint):
    ap = appoint[['user_id', 'clue_id']]
    ap = ap.groupby('user_id').groups
    clue_pairs = []
    count = 0
    for pair in user_pairs:
        for clue1 in ap[pair[0]]:
            for clue2 in ap[pair[1]]:
                clue_pairs.append([clue1, clue2])
                count = count + 1
    print count
    clue_pairs = pd.DataFrame(clue_pairs, columns = ['clue_id', 'clue_id_2'])
    return clue_pairs

def calculate_transformation_probability(pairs2, car2, feature2):
    pairs2 = pairs2.join(car2.set_index('clue_id')[feature2], on = 'clue_id').dropna()
    pairs2 = pairs2.join(car2.set_index('clue_id')[feature2], on = 'clue_id_2', rsuffix = '_2').dropna()
    feature_2 = feature2 + '_2'
    transformationProbability2 = pd.crosstab(pairs2[feature2], pairs2[feature_2])
    for i in range(len(transformationProbability2)):
        transformationProbability2.iloc[i] = transformationProbability2.iloc[i] / transformationProbability2.iloc[i].max()
    transformationProbability2 = transformationProbability2.round(3)
    return transformationProbability2

def getFileName(feature3):
    return '../../data/categoryTransFormationProbability/' + feature3 + 'NoDealerNegativeSample.tsv'

def writeToFile(filename1, transformationProbability1):
    transformationProbability1.to_csv(filename1, sep = '\t')

def calculateAllTransformationProbability(appoint4, car4, catgoryFeatures4):
    newAppoint4 = getNewAppoint(appoint4)
    print 'pairing......'
    user_pairs = get_user_pairs(newAppoint4, car4)
    clue_pairs = get_clue_pairs(user_pairs, newAppoint4)
    for feature4 in catgoryFeatures4:
        print feature4
        newCar4 = getNewCar(feature4, car4)
        print 'calculate transformation Probability...'
        transformationProbability4 = calculate_transformation_probability(clue_pairs, newCar4, feature4)
        fileName4 = getFileName(feature4)
        writeToFile(fileName4, transformationProbability4)
        print 'done!'

if __name__=="__main__":
      catgoryFeatures = ['emission_standard', 'car_year', 'transfer_num', 'minor_category_id', 'source_level', 'auto_type', 'carriages', 'fuel_type', 'gearbox', 'car_color', 'guobie', 'seats']
      carFile = '../../data/hl_car.tsv'
      appointFile = '../../data/hl_appoint.tsv'
      car = readFile(carFile)
      appoint = readFile(appointFile)
      appointNoDealer = getAppointNoDealer(appoint)
      calculateAllTransformationProbability(appointNoDealer, car,  catgoryFeatures)

