import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
from scipy.spatial.distance import *
import math
from scipy.special import rel_entr, entr
from scipy.stats import entropy
import numpy as np

THEATA = 0.02

def fun(x):
    if x > 1.0:
        return 1.0 / x
    else:
        return x

def calc_ent(x):
    '''
    calculate shanno ent of x
    '''
    x_value_list = set(x.unique())
    ent = 0.0
    for x_value in x_value_list:
        p = float(len(x[x == x_value])) / len(x)
        logp = np.log2(p)
        ent -= p * logp
    return ent

def calc_condition_ent(x, y):
    '''
    calculate ent H(y|x)
    '''
    x_value_list = set(x.unique())
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(len(sub_y)) / len(y)) * temp_ent
    return ent

def calc_ent_grap(x,y):
    '''
    calculate ent grap
    '''
    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent
    return ent_grap
'''
if __name__ =="__main__":
    old = pd.read_csv('../../data/testData/test1.tsv', sep = '\t')
    car = pd.read_csv('../../data/hl_car.tsv', sep = '\t')
    num_vars = ['suggest_price', 'base_price', 'price', 'road_haul', 'air_displacement']
    cat_vars = ['plate_city_id', 'city_id', 'guobie', 'minor_category_id', 'tag_id', 'auto_type', 'fuel_type', 'car_color', 'emission_standard', 'car_year', 'gearbox', 'carriages', 'seats']
'''
class add_features:
    def __init__(self, old_features, car, num_vars, cat_vars):
        self.old = old_features
        self.car = car
        self.num_vars = num_vars
        self.cat_vars = cat_vars

    def num_vars_normalization(self):
        for feature in self.num_vars:
            pairs = self.old[['clue_id', 'clue_id_2']]
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id')
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
            feature_2 = feature + '_2'
            new = feature + '_4'
            new1 = feature + '_5'
            pairs[new] = 1.0 * abs(pairs[feature] - pairs[feature_2]) / (pairs[feature] + pairs[feature_2])
            pairs[new1] = 1.0 * abs(pairs[feature] - pairs[feature_2]) / (pairs[feature].max() - pairs[feature_2].min())
            self.old = pd.concat([self.old, pairs[new]], axis=1)
            self.old = pd.concat([self.old, pairs[new1]], axis=1)
            del(pairs)
    '''
    def eight_distances_of_num_vars(self):
        pairs = self.old[['clue_id', 'clue_id_2']]
        for feature in self.num_vars:
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id')
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
            feature_2 = feature + '_2'
            pairs[feature] = 1.0 * pairs[feature] / pairs[feature].max()
            pairs[feature_2] = 1.0 * pairs[feature_2] / pairs[feature_2].max()
        num_vars_2 = [feature + '_2' for feature in self.num_vars]
        mp = pairs[self.num_vars].as_matrix()
        mp2 = pairs[num_vars_2].as_matrix()
        a = []
        for i in range(len(mp)):
            a.append([cosine(mp[i], mp2[i]), euclidean(mp[i], mp2[i]), braycurtis(mp[i], mp2[i]), canberra(mp[i], mp2[i]), chebyshev(mp[i], mp2[i]), cityblock(mp[i], mp2[i]), correlation(mp[i], mp2[i]), sqeuclidean(mp[i], mp2[i])])
        a = pd.DataFrame(a, columns = ['distance1', 'ditance2', 'distance3', 'distance4', 'distance5', 'ditance6', 'distance7', 'distance8'])
        self.old = pd.concat([self.old, a], axis = 1)

    def eight_distances_of_cat_vars(self):
        for feature in self.cat_vars:
            new_feature = [feature] + self.num_vars
            new_car = self.car[new_feature]
            m = new_car.groupby(feature).mean()
            pairs = self.old[['clue_id', 'clue_id_2']]
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id')
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
            feature_2 = feature + '_2'
            pairs = pairs.join(m, on = feature)
            pairs = pairs.join(m, on = feature_2, rsuffix = '_2')
            num_vars_2 = [featue + '_2' for featue in self.num_vars]
            mp = pairs[self.num_vars].as_matrix()
            mp2 = pairs[num_vars_2].as_matrix()
            a = []
            for i in range(len(mp)):
                a.append([cosine(mp[i], mp2[i]), euclidean(mp[i], mp2[i]), braycurtis(mp[i], mp2[i]), canberra(mp[i], mp2[i]), chebyshev(mp[i], mp2[i]), cityblock(mp[i], mp2[i]), correlation(mp[i], mp2[i]), sqeuclidean(mp[i], mp2[i])])
            fs = ['distance1', 'ditance2', 'distance3', 'distance4', 'distance5', 'ditance6', 'distance7', 'distance8']
            cols = [feature + '_' + x for x in fs]
            a = pd.DataFrame(a, columns = cols)
            self.old = pd.concat([self.old, a], axis = 1)
            for ft in self.num_vars:
                ft_2 = ft + '_2'
                nf1 = feature + '_' + ft + '_1'
                pairs[nf1] = 1.0 * abs(pairs[ft] - pairs[ft_2]) / (pairs[ft].max() - pairs[ft].min())
                nf2 = feature + '_' + ft + '_2'
                pairs[nf2] = 1.0 * pairs[ft] / pairs[ft_2]
                pairs[nf2] = pairs[nf2].apply(fun)
                self.old = pd.concat([self.old, pairs[nf1]], axis=1)
                self.old = pd.concat([self.old, pairs[nf2]], axis=1)
            del(pairs)
    '''

    def SO_distance(self):
        dis = 'SO_distance'
        pairs = self.old[['clue_id', 'clue_id_2']]
        pairs[dis] = 0
        for feature in self.cat_vars:
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id')
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
            new_feature = feature + '_SO'
            feature_2 = feature + '_2'
            new_feature_2 = feature + 'SO_2'
            Sx = 100 / (self.car[feature].max() - self.car[feature].min())
            Ox = -self.car[feature].min()
            pairs[new_feature] = pairs[feature] * Sx + Ox
            pairs[new_feature_2] = pairs[feature_2] * Sx + Ox
            pairs[dis] += (pairs[new_feature] - pairs[new_feature_2]) ** 2
        self.old = pd.concat([self.old, pairs[dis]], axis=1)
        #print self.old[dis]
        del(pairs)
        self.old[dis] = self.old[dis].apply(math.sqrt)
        #print self.old[dis]

    def KL_distance(self):
        dis = 'KL_distance'
        pairs = self.old[['clue_id', 'clue_id_2']]
        pairs[dis] = 0
        for feature in self.cat_vars:
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id')
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
            new_feature = feature + '_KL'
            feature_2 = feature + '_2'
            new_feature_2 = feature + 'KL_2'
            sm = self.car[feature].unique().sum()
            pairs[new_feature] = 1.0 * pairs[feature] / sm
            pairs[new_feature_2] = 1.0 * pairs[feature_2] / sm
            pairs[dis] += rel_entr(pairs[new_feature], pairs[new_feature_2]) + rel_entr(pairs[new_feature_2], pairs[new_feature])
        self.old = pd.concat([self.old, pairs[dis]], axis=1)
        del(pairs)
        self.old[dis].replace([np.inf, -np.inf], 0, inplace = True)

    def Overlap(self):
        dis = 'Overlap'
        pairs = self.old[['clue_id', 'clue_id_2']]
        pairs[dis] = 0
        w = 1.0 / len(self.cat_vars)
        for feature in self.cat_vars:
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id')
            pairs = pairs.join(self.car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
            feature_2 = feature + '_2'
            if pairs[feature] == pairs[feature_2]:
                pairs[dis] += 1
        self.old = pd.concat([self.old, pairs[dis]], axis=1)
        del(pairs)
        self.old[dis].replace([np.inf, -np.inf], 0, inplace = True)

    def run(self):
        self.num_vars_normalization()
        self.eight_distances_of_num_vars()
        self.eight_distances_of_cat_vars()
        self.SO_distance()
        self.KL_distance()
        return self.old

    '''
    dis = 'Con_distance'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs = pairs.join(car.set_index('clue_id')[cat_vars], on = 'clue_id')
    pairs = pairs.join(car.set_index('clue_id')[cat_vars], on = 'clue_id_2', rsuffix = '_2')
    for feature in cat_vars:
        contextX = []
        cors = {}
        for feature1 in cat_vars:
            IG = calc_ent_grap(pairs[feature1], pairs[feature])
            cor = 1.0 * IG / calc_ent(pairs[feature])
            print cor
            if cmp(cor,THEATA):
                contextX.append(feature1)
                cors[feature1] = cor
        print len(contextX)
        impactX = {}
        for feature1 in contextX:
            impactX[feature1] = cors[feature1] * [(1 - 0.5 * cors[feature1]) ** 2]
            sm = 0
            y_value_list = set(pairs[feature1].unique())
            for y_value in y_value_list:

    print old
    old.to_csv('test3.tsv', index = False, sep = '\t')
    '''
