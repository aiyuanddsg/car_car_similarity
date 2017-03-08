import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
import random as rd

class negative_sample_transformation_probability:

    def __init__(self, appoint_file_name, car_file_name, features):
        self.appoint_file_name = appoint_file_name
        self.car_file_name = car_file_name
        self.features = features

    def readFile(self, filename):
        fileContent = pd.read_csv(filename, sep = '\t')
        return fileContent

    def getNewAppoint(self, appoint):
        columns = ['clue_id', 'create_time', 'user_id']
        tempAppoint = appoint[columns].dropna()
        return tempAppoint

    def getNewCar(self, feature, car):
        newCar = car[['clue_id', feature]]
        newCar = newCar[newCar[feature] > 0]
        return newCar

    def get_user_pairs(self, appoint, car):
        ap = appoint.join(car.set_index('clue_id')['price'], on = 'clue_id')
        ap = ap[['user_id', 'price']]
        mean_price = ap.groupby('user_id').mean()
        mean_price = mean_price[mean_price['price'] > 0]
        amp = mean_price.sort_values('price', ascending=True)
        idx = amp.index
        user_pairs = []
        count = 0
        for i in range(0, 1000):
            j = rd.randint(0, len(idx) / 2)
            k = rd.randint(len(idx) / 2 + 1, len(idx) - 1)
            if amp.ix[idx[j]]['price'] * 4 < amp.ix[idx[k]]['price']:
                if rd.random() < 0.5:
                    user_pairs.append([idx[j], idx[k]])
                else:
                    user_pairs.append([idx[k], idx[j]])
        return user_pairs

    def get_clue_pairs(self, user_pairs, appoint, car):
        ap = appoint[['user_id', 'clue_id']]
        ap = ap.groupby('user_id').groups
        clue_pairs = []
        for pair in user_pairs:
            for clue1 in ap[pair[0]]:
                count = 0
                for clue2 in ap[pair[1]]:
                    count = count + 1
                    if count > 2:
                        continue
                    clue_pairs.append([appoint.iloc[clue1]['clue_id'], appoint.iloc[clue2]['clue_id']])
        clue_pairs = pd.DataFrame(clue_pairs, columns = ['clue_id', 'clue_id_2'])
        return clue_pairs

    def calculate_transformation_probability(self, pairs, car, feature):
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id').dropna()
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2').dropna()
        feature_2 = feature + '_2'
        transformationProbability = pd.crosstab(pairs[feature], pairs[feature_2])
        for i in range(len(transformationProbability)):
            transformationProbability.iloc[i] = transformationProbability.iloc[i] / transformationProbability.iloc[i].max()
        transformationProbability = transformationProbability.round(3)
        return transformationProbability

    def calculateAllTransformationProbability(self, appoint, car, catgoryFeatures):
        newAppoint = self.getNewAppoint(appoint)
        user_pairs = self.get_user_pairs(newAppoint, car)
        clue_pairs = self.get_clue_pairs(user_pairs, newAppoint, car)
        transformationProbability = {}
        '''
        for feature in catgoryFeatures:
            newCar = self.getNewCar(feature, car)
            transformationProbability[feature] = self.calculate_transformation_probability(clue_pairs, newCar, feature)
        '''
        return clue_pairs, transformationProbability

    def clue_pairs_correct(self, clue_pairs, car):
        clues = list(car['clue_id'])
        count = 0
        print len(clue_pairs)
        for i in range(len(clue_pairs)):
            if clue_pairs.iloc[i]['clue_id'] in clues and clue_pairs.iloc[i]['clue_id_2'] in clues:
                count = count + 1
        print count

    def run(self):
        car = self.readFile(self.car_file_name)
        appoint = self.readFile(self.appoint_file_name)
        pairs, tp = self.calculateAllTransformationProbability(appoint, car, self.features)
        self.clue_pairs_correct(pairs, car)

if __name__ =="__main__":

    features = ['emission_standard', 'car_year', 'transfer_num', 'minor_category_id', 'source_level', 'auto_type', 'carriages', 'fuel_type', 'gearbox', 'car_color', 'guobie', 'seats']
    car_file = '../../data/hl_car.tsv'
    appoint_file = '../../data/hl_appoint.tsv'

    nstp = negative_sample_transformation_probability(appoint_file, car_file, features)
    nstp.run()
