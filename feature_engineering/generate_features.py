import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd

class generate_features:

    def __init__(self, file_name, new_pairs, no_dealer_transformation_probability, features, numerical_features, time_features):
        self.file_name = file_name
        self.new_pairs = new_pairs
        self.no_dealer_transformation_probability = no_dealer_transformation_probability
        self.features = features
        self.numerical_features = numerical_features
        self.time_features = time_features

    def readFile(self, file_name):
        fileContent = pd.read_csv(file_name, sep = '\t')
        return fileContent

    def generate_input(self, pairs, features, transformation_probability, car, numerical_features, time_features):
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
            similarity.append(pairs.iloc[i]['LABEL'])
            for feature in features:
                clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id']][feature]
                the_other_clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id_2']][feature]

                if clue_feature_value <= 0 or the_other_clue_feature_value <= 0:
                    tp = 0
                else:
                    if clue_feature_value not in transformation_probability[feature].index or the_other_clue_feature_value not in transformation_probability[feature]:
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
                if distance > 1.0:
                    distance = 1.0 / distance
                similarity.append(distance)

            for feature in numerical_features:
                clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id']][feature]
                the_other_clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id_2']][feature]
                if the_other_clue_feature_value != 0:
                    distance = 1.0 * (clue_feature_value - the_other_clue_feature_value) / the_other_clue_feature_value
                else:
                    distance = 0.0
                if distance < 0.0:
                    distance = -distance
                similarity.append(distance)

            for feature in numerical_features:
                clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id']][feature]
                the_other_clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id_2']][feature]
                if clue_feature_value != 0:
                    distance = 1.0 * (clue_feature_value - the_other_clue_feature_value) / clue_feature_value
                else:
                    distance = 0.0
                if distance < 0.0:
                    distance = -distance
                similarity.append(distance)
            '''
            for feature in numerical_features:
                clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id']][feature]
                the_other_clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id_2']][feature]
                if the_other_clue_feature_value != 0:
                    distance = 1.0 * (clue_feature_value - the_other_clue_feature_value)
                else:
                    distance = 0.0
                if distance < 0.0:
                    distance = -distance
                similarity.append(distance)
            '''

            for feature in numerical_features:
                clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id']][feature]
                the_other_clue_feature_value = tempCar.ix[pairs.iloc[i]['clue_id_2']][feature]
                if the_other_clue_feature_value != 0 and clue_feature_value != 0:
                    distance = 1.0 * (clue_feature_value - the_other_clue_feature_value) * (clue_feature_value + the_other_clue_feature_value)
                else:
                    distance = 0.0
                if distance < 0.0:
                    distance = -distance
                similarity.append(distance)

            for feature in time_features:
                clue1_year = tempCar.ix[pairs.iloc[i]['clue_id']][feature[0]]
                clue1_month = tempCar.ix[pairs.iloc[i]['clue_id']][feature[1]]
                clue2_year = tempCar.ix[pairs.iloc[i]['clue_id_2']][feature[0]]
                clue2_month = tempCar.ix[pairs.iloc[i]['clue_id_2']][feature[1]]
                years = clue1_year - clue2_year
                months = years * 12 + clue1_month - clue2_month
                if months < 0:
                    months = -months
                months = months + 1
                months = 1.0 / months
                similarity.append(months)

            if len(similarity) > 1:
                similaritys.append(similarity)
        similaritys=pd.DataFrame(similaritys)
        return similaritys

    def get_input_file_name(self):
        return 'test1.tsv'

    def write_input(self, file_name, no_dealer_input, features):
        cat_vars = ['plate_city_id', 'city_id', 'guobie', 'minor_category_id', 'tag_id', 'car_id', 'auto_type', 'fuel_type', 'car_color', 'emission_standard', 'car_year_t', 'gearbox_t', 'carriages_t', 'seats_t']
        num_vars = ['suggest_price', 'base_price', 'price', 'road_haul', 'car_year', 'air_displacement', 'gearbox', 'carriages', 'seats']
        num_vars1 = [ft + '_1' for ft in num_vars]
        num_vars2 = [ft + '_2' for ft in num_vars]
        num_vars3 = [ft + '_3' for ft in num_vars]
        time_vars = ['license', 'business', 'strong', 'audit']
        fts =['clue_id', 'clue_id_2', 'LABEL'] + cat_vars + num_vars + num_vars1 + num_vars2 + num_vars3 + time_vars
        no_dealer_input.columns=fts
        return no_dealer_input

    def run(self):
        car = self.readFile(self.file_name)
        no_dealer_input = self.generate_input(self.new_pairs, self.features, self.no_dealer_transformation_probability, car, self.numerical_features, self.time_features)
        ofile_name = self.get_input_file_name()
        fts = self.features + self.numerical_features
        return self.write_input(ofile_name, no_dealer_input, fts)
