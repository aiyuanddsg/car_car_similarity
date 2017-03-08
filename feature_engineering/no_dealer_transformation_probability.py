#!/usr/bin/env python
# encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd

class no_dealer_transformation_probability:

    def __init__(self, appoint_file_name, car_file_name, features):
        self.appoint_file_name = appoint_file_name
        self.car_file_name = car_file_name
        self.features = features

    def readFile(self, file_name):
        fileContent = pd.read_csv(file_name, sep = '\t')
        return fileContent

    def getAppointNoDealer(self, appoint):
        appoint = appoint[appoint['dealer'] == 0]
        return appoint

    def getNewAppoint(self, appoint):
        columns = ['clue_id', 'create_time', 'user_id']
        tempAppoint = appoint[columns].dropna()
        return tempAppoint

    def getNewCar(self, feature, car):
        newCar = car[['clue_id', feature]]
        newCar = newCar[newCar[feature] > 0]
        return newCar

    def getCluePairs(self, appoint):
        groups = appoint.sort_values('create_time').groupby('user_id', sort = False)
        pairs = []
        for idx, g in groups:
            clues = list(g['clue_id'])
            for i in range(len(clues) - 1):
                for j in range(i + 1, len(clues)):
                    pairs.append([clues[i], clues[j]])
        pairs = pd.DataFrame(pairs, columns=['clue_id','clue_id_2'])
        return pairs

    def calculateTransformationProbability(self, pairs, car, feature):
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id').dropna()
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2').dropna()
        feature_2 = feature + '_2'
        transformationProbability = pd.crosstab(pairs[feature], pairs[feature_2])
        #pairs[[feature, feature_2]].to_csv(feature, index=False, sep='\t')
        for i in range(len(transformationProbability)):
            transformationProbability.iloc[i] = transformationProbability.iloc[i] / transformationProbability.iloc[i].max()
        #transformationProbability = transformationProbability.round(3)
        return transformationProbability

    def calculateAllTransformationProbability(self, appoint, car, catgoryFeatures):
        newAppoint = self.getNewAppoint(appoint)
        pairs = self.getCluePairs(newAppoint)

        transformationProbability = {}
        for feature in catgoryFeatures:
            newCar = self.getNewCar(feature, car)
            transformationProbability[feature] = self.calculateTransformationProbability(pairs, newCar, feature)
        return pairs, transformationProbability

    def run(self):
        car = self.readFile(self.car_file_name)
        appoint = self.readFile(self.appoint_file_name)

        #appoint = appoint.sample(n = 50000, replace=True)

        appointNoDealer = self.getAppointNoDealer(appoint)
        pairs, tp = self.calculateAllTransformationProbability(appointNoDealer, car, self.features)
        return pairs, tp
