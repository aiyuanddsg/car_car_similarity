import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
import random as rd

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
        for i in range(len(transformationProbability)):
            transformationProbability.iloc[i] = transformationProbability.iloc[i] / transformationProbability.iloc[i].max()
        transformationProbability = transformationProbability.round(3)
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

        #appoint = appoint.sample(n = 5000, replace=True)

        appointNoDealer = self.getAppointNoDealer(appoint)
        pairs, tp = self.calculateAllTransformationProbability(appointNoDealer, car, self.features)
        return pairs, tp

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
        for i in range(0, 10):
            j = rd.randint(0, len(idx) / 2)
            k = rd.randint(len(idx) / 2 + 1, len(idx) - 1)
            if amp.ix[idx[j]]['price'] * 3 < amp.ix[idx[k]]['price']:
                if rd.random() < 0.5:
                    user_pairs.append([idx[j], idx[k]])
                else:
                    user_pairs.append([idx[k], idx[j]])
        return user_pairs

    def get_clue_pairs(self, user_pairs, appoint):
        app = appoint[['user_id', 'clue_id']]
        ap = app.groupby('user_id').groups
        clue_pairs = []
        for pair in user_pairs:
            for clue1 in ap[pair[0]]:
                count = 0
                for clue2 in ap[pair[1]]:
                    count = count + 1
                    if count > 2:
                        continue
                    clue_pairs.append([app.iloc[clue1]['clue_id'], app.iloc[clue2]['clue_id']])
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
        clue_pairs = self.get_clue_pairs(user_pairs, newAppoint)
        transformationProbability = {}
        for feature in catgoryFeatures:
            newCar = self.getNewCar(feature, car)
            transformationProbability[feature] = self.calculate_transformation_probability(clue_pairs, newCar, feature)
        return clue_pairs, transformationProbability

    def run(self):
        car = self.readFile(self.car_file_name)
        appoint = self.readFile(self.appoint_file_name)
        pairs, tp = self.calculateAllTransformationProbability(appoint, car, self.features)
        return pairs, tp

class generate_LR_input:

    def __init__(self, file_name, no_dealer_pairs, no_dealer_transformation_probability, features, numerical_features):
        self.file_name = file_name
        self.no_dealer_pairs = no_dealer_pairs
        self.no_dealer_transformation_probability = no_dealer_transformation_probability
        self.features = features
        self.numerical_features = numerical_features

    def readFile(self, file_name):
        fileContent = pd.read_csv(file_name, sep = '\t')
        return fileContent

    def generate_input(self, pairs, features, transformation_probability, car, numerical_features):
        tempCar = car.set_index('clue_id')
        similaritys = []
        count = 0
        for i in range(len(pairs)):
            count = count + 1
            if count % 1000 == 0:
                print str(count) + ' / ' + str(len(pairs))

            if pairs.iloc[i]['clue_id'] not in tempCar.index or pairs.iloc[i]['clue_id_2'] not in tempCar.index:
                continue

            similarity = []
            similarity.append(pairs.iloc[i]['clue_id'])
            similarity.append(pairs.iloc[i]['clue_id_2'])
            for feature in features:
                clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id']][feature]
                the_other_clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id_2']][feature]

                if clue_feature_value <= 0 or the_other_clue_feature_value <= 0:
                    tp = 0
                else:
                    tp = transformation_probability[feature].ix[clue_feature_value][the_other_clue_feature_value]
                similarity.append(tp)

            for feature in numerical_features:
                clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id']][feature]
                the_other_clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id_2']][feature]
                if clue_feature_value > 0:
                    distance = 1.0 * the_other_clue_feature_value / clue_feature_value
                else:
                    distance = 0.0
                if distance < 1 and distance != 0:
                    distance = 1 / distance
                distance = round(distance, 3)
                similarity.append(distance)

            if len(similarity) > 0:
                similaritys.append(similarity)
        return similaritys

    def get_input_file_name(self):
        return 'positive_samples2.tsv'

    def write_input(self, file_name, no_dealer_input):
        features = ['clue_id', 'clue_id_2', 'emission_standard', 'car_year', 'transfer_num', 'minor_category_id', 'car_id', 'tag_id', 'source_level', 'auto_type', 'carriages', 'fuel_type', 'gearbox', 'car_color', 'guobie', 'seats', 'price']
        no_dealer_input=pd.DataFrame(no_dealer_input, columns=features)
        no_dealer_input.to_csv(file_name, index=False, sep='\t')

    def run(self):
        car = self.readFile(self.file_name)
        no_dealer_input = self.generate_input(self.no_dealer_pairs, self.features, self.no_dealer_transformation_probability, car, self.numerical_features)
        ofile_name = self.get_input_file_name()
        self.write_input(ofile_name, no_dealer_input)

if __name__ =="__main__":

    features = ['emission_standard', 'car_year', 'transfer_num', 'minor_category_id', 'car_id', 'tag_id', 'source_level', 'auto_type', 'carriages', 'fuel_type', 'gearbox', 'car_color', 'guobie', 'seats']
   # features = ['emission_standard', 'car_year', 'transfer_num', 'minor_category_id', 'car_id', 'tag_id', 'source_level', 'auto_type', 'carriages', 'fuel_type', 'gearbox', 'car_color', 'guobie', 'seats']
    #numerical_features = ['price', 'base_price', 'suggest_price']
    numerical_features = ['price']
    car_file = '../../data/hl_car.tsv'
    appoint_file = '../../data/hl_appoint.tsv'

    print 'no dealer'
    ndtp = no_dealer_transformation_probability(appoint_file, car_file, features)
    nd_pairs, nd_tp = ndtp.run()
    print len(nd_pairs)
    '''
    print 'negative sample'
    nstp = negative_sample_transformation_probability(appoint_file, car_file, features)
    ns_pairs, ns_tp = nstp.run()
    '''
    print 'LR'
    lri = generate_LR_input(car_file, nd_pairs, nd_tp, features, numerical_features)
    lri.run()
