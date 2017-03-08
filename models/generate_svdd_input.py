import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
import random as rd

class transformation_probability:

    def __init__(self, appoint_file_name, car_file_name, features, dealer_type):
        self.appoint_file_name = appoint_file_name
        self.car_file_name = car_file_name
        self.features = features
        self.dealer_type = dealer_type

    def readFile(self, file_name):
        fileContent = pd.read_csv(file_name, sep = '\t')
        return fileContent

    def getAppointNoDealer(self, appoint, dealer_type):
        if dealer_type == True:
            appoint = appoint[appoint['dealer'] == 0]
        else:
            appoint = appoint[appoint['dealer'] != 0]
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
                j = i + 1
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

        #appoint = appoint.sample(n = 50000, replace=True)

        appointNoDealer = self.getAppointNoDealer(appoint, self.dealer_type)
        pairs, tp = self.calculateAllTransformationProbability(appointNoDealer, car, self.features)
        return pairs, tp

class generate_LR_input:

    def __init__(self, file_name, no_dealer_pairs, negative_sample_pairs, no_dealer_transformation_probability, negative_sample_transformation_probability, features, numerical_features):
        self.file_name = file_name
        self.no_dealer_pairs = no_dealer_pairs
        self.negative_sample_pairs = negative_sample_pairs
        self.no_dealer_transformation_probability = no_dealer_transformation_probability
        self.negative_sample_transformation_probability = negative_sample_transformation_probability
        self.features = features
        self.numerical_features = numerical_features

    def readFile(self, file_name):
        fileContent = pd.read_csv(file_name, sep = '\t')
        return fileContent

    def generate_input(self, pairs, features, numerical_features, transformation_probability, car):
        tempCar = car.set_index('clue_id')
        print tempCar.index
        similaritys = []
        count = 0
        for i in range(len(pairs)):
            count = count + 1
            if count % 1000 == 0:
                print str(count) + ' / ' + str(len(pairs))

            if pairs.iloc[i]['clue_id'] not in tempCar.index or pairs.iloc[i]['clue_id_2'] not in tempCar.index:
                continue

            similarity = []
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
        return '../../data/categorySvddInput/input.csv'

    def write_input(self, file_name, no_dealer_input, all_input):
        f = open(file_name, "w")
        for i in range(len(no_dealer_input)):
            f.write(str(1))
            count = 0
            for j in range(0, len(no_dealer_input[i])):
                f.write(' ' + str(count) + ':' + str(no_dealer_input[i][j]))
                count = count + 1
            f.write('\n')
        for i in range(len(all_input)):
            f.write(str(0))
            count = 0
            for j in range(0, len(all_input[i])):
                f.write(' ' + str(count) + ':' + str(all_input[i][j]))
                count = count + 1
            f.write('\n')
        f.close()

    def run(self):
        car = self.readFile(self.file_name)
        print 'no dealer...'
        no_dealer_input = self.generate_input(self.no_dealer_pairs, self.features, self.numerical_features, self.no_dealer_transformation_probability, car)
        print 'negative dealer...'
        negative_sample_input = self.generate_input(self.negative_sample_pairs, self.features, self.numerical_features, self.negative_sample_transformation_probability, car)
        ofile_name = self.get_input_file_name()
        print 'write...'
        self.write_input(ofile_name, no_dealer_input, negative_sample_input)

if __name__ =="__main__":

    features = ['emission_standard', 'car_year', 'transfer_num', 'minor_category_id', 'source_level', 'auto_type', 'carriages', 'fuel_type', 'gearbox', 'car_color', 'guobie', 'seats']
    numerical_features = ['price']
    car_file = '../../data/hl_car.tsv'
    appoint_file = '../../data/hl_appoint.tsv'

    print 'no dealer'
    ndtp = transformation_probability(appoint_file, car_file, features, True)
    nd_pairs, nd_tp = ndtp.run()
    print len(nd_pairs)

    n_tp = transformation_probability(appoint_file, car_file, features, False)
    n_pairs, n_tp = n_tp.run()


    print 'LR'
    glri = generate_LR_input(car_file, nd_pairs, n_pairs, nd_tp, n_tp, features, numerical_features)
    glri.run()
