import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd

class generate_features:

    def __init__(self, file_name, new_pairs, no_dealer_transformation_probability, features, numerical_features, time_features, car):
        self.file_name = file_name
        self.new_pairs = new_pairs
        self.no_dealer_transformation_probability = no_dealer_transformation_probability
        self.features = features
        self.numerical_features = numerical_features
        self.time_features = time_features
        self.car = car

    def readFile(self, file_name):
        fileContent = pd.read_csv(file_name, sep = '\t')
        return fileContent

    def trans(self, x):
        result = 0
        if x[0] not in self.car.set_index('clue_id').index or x[1] not in self.car.set_index('clue_id').index:
            result = 0
        else:
            clue_feature_value = self.car.ix[x[0]][x[2]]
            the_other_clue_feature_value = self.car.ix[x[1]][x[2]]
            if clue_feature_value <= 0 or the_other_clue_feature_value <= 0:
                result = 0
            else:
                if clue_feature_value not in self.no_dealer_transformation_probability[x[2]].index or the_other_clue_feature_value not in self.no_dealer_transformation_probability[feature]:
                    result = 0
                else:
                    result = self.no_dealer_transformation_probability[x[2]].ix[clue_feature_value][the_other_clue_feature_value]
        return result

    def generate_trans(self, pairs, features, transformation_probability):
        for feature in features:
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id')
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')

        trans_features = [feature + '_trans' for feature in features]
        for feature in features:
            feature_2 = feature + '_2'
            pairs_temp = pairs[[feature, feature_2]]
            pairs_temp.columns = [0,1]
            pairs_temp[3] = feature
            trans_feature = feature + '_trans'
            pairs_temp[trans_feature] = pairs_temp.apply(self.trans, axis=1)
        print pairs_temp[trans_features]
        return pairs_temp[trans_features]

    def rio(self, x):
        distance = 0
        if x[0] not in self.car.set_index('clue_id').index or x[1] not in self.car.set_index('clue_id').index:
            distance = 0
        else:
            clue_feature_value = self.car.ix[x[0]][x[2]]
            the_other_clue_feature_value = self.car.ix[x[1]][x[2]]
            if clue_feature_value > 0:
                distance = 1.0 * the_other_clue_feature_value / clue_feature_value
            else:
                distance = 0.0
            if distance > 1.0:
                distance = 1.0 / distance
        return distance

    def generate_rio(self, pairs, numerical_features):
        rio_features = [feature + '_rio' for feature in numerical_features]
        for feature in numerical_features:
            feature_2 = feature + '_2'
            pairs_temp = pairs[[feature, feature_2]]
            pairs_temp.columns = [0,1]
            pairs_temp[3] = feature
            rio_feature = feature + '_rio'
            pairs_temp[rio_feature] = pairs_temp.apply(self.trans, axis=1)
        print pairs_temp[rio_features]
        return pairs_temp[rio_features]

    def sub_rio(self, x):
        distance = 0
        if x[0] not in self.car.set_index('clue_id').index or x[1] not in self.car.set_index('clue_id').index:
            distance = 0
        else:
            clue_feature_value = self.car.ix[x[0]][x[2]]
            the_other_clue_feature_value = self.car.ix[x[1]][x[2]]
            if the_other_clue_feature_value != 0:
                distance = 1.0 * (clue_feature_value - the_other_clue_feature_value) / the_other_clue_feature_value
            else:
                distance = 0.0
            if distance < 0:
                distance = -distance
        return distance

    def generate_sub_rio(self, pairs, numerical_features):
        sub_rio_features = [feature + '_sub_rio' for feature in numerical_features]
        for feature in numerical_features:
            feature_2 = feature + '_2'
            pairs_temp = pairs[[feature, feature_2]]
            pairs_temp.columns = [0,1]
            pairs_temp[3] = feature
            sub_rio_feature = feature + '_sub_rio'
            pairs_temp[sub_rio_feature] = pairs_temp.apply(self.trans, axis=1)
        print pairs_temp[sub_rio_features]
        return pairs_temp[sub_rio_features]

    def get_input_file_name(self):
        return 'test1.tsv'

    def write_input(self, file_name, input_file):
        input_file.to_csv(file_name, index = False, sep = '\t')
        return input_file

    def run(self):
        trans = self.generate_trans(self.new_pairs, self.features, self.no_dealer_transformation_probability)
        rio = self.generate_rio(self.new_pairs, self.numerical_features)
        sub_rio = self.generate_sub_rio(self.new_pairs, self.numerical_features)
        input_file = pd.concat([self.new_pairs, trans], axis=1)
        input_file = pd.concat([input_file, rio], axis=1)
        input_file = pd.concat([input_file, sub_rio], axis=1)
        ofile_name = self.get_input_file_name()
        return self.write_input(ofile_name, input_file)
