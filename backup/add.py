import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
from scipy.spatial.distance import *
import numpy as np

def fun(x):
    if x > 1.0:
        return 1.0 / x
    else:
        return x

if __name__ =="__main__":
    old = pd.read_csv('test1.tsv', sep = '\t')
    car = pd.read_csv('../hl_car.tsv', sep = '\t')
    num_vars = ['suggest_price', 'base_price', 'price', 'road_haul', 'air_displacement']
    cat_vars = ['plate_city_id', 'city_id', 'guobie', 'minor_category_id', 'tag_id', 'auto_type', 'fuel_type', 'car_color', 'emission_standard', 'car_year', 'gearbox', 'carriages', 'seats']
    #t = add_feature(old, car, num_vars)
    for feature in num_vars:
        #print feature
        pairs = old[['clue_id', 'clue_id_2']]
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        new = feature + '_4'
        new1 = feature + '_5'
        pairs[new] = 1.0 * abs(pairs[feature] - pairs[feature_2]) / (pairs[feature] + pairs[feature_2])
        pairs[new1] = 1.0 * abs(pairs[feature] - pairs[feature_2]) / (pairs[feature].max() - pairs[feature_2].min())
        old = pd.concat([old, pairs[new]], axis=1)
        old = pd.concat([old, pairs[new1]], axis=1)
        del(pairs)

    for feature in cat_vars:
        #print feature
        new_feature = [feature] + num_vars
        new_car = car[new_feature]
        m = new_car.groupby(feature).mean()
        pairs = old[['clue_id', 'clue_id_2']]
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        pairs = pairs.join(m, on = feature)
        pairs = pairs.join(m, on = feature_2, rsuffix = '_2')
        #print pairs
        num_vars_2 = [featue + '_2' for featue in num_vars]
        mp = pairs[num_vars].as_matrix()
        mp2 = pairs[num_vars_2].as_matrix()
        a = []
        for i in range(len(mp)):
            a.append([cosine(mp[i], mp2[i]), euclidean(mp[i], mp2[i]), braycurtis(mp[i], mp2[i]), canberra(mp[i], mp2[i]), chebyshev(mp[i], mp2[i]), cityblock(mp[i], mp2[i]), correlation(mp[i], mp2[i]), sqeuclidean(mp[i], mp2[i])])
        fs = ['distance1', 'ditance2', 'distance3', 'distance4', 'distance5', 'ditance6', 'distance7', 'distance8']
        cols = [feature + '_' + x for x in fs]
        a = pd.DataFrame(a, columns = cols)
        old = pd.concat([old, a], axis = 1)
        for ft in num_vars:
            ft_2 = ft + '_2'
            nf1 = feature + '_' + ft + '_1'
            pairs[nf1] = 1.0 * abs(pairs[ft] - pairs[ft_2]) / (pairs[ft].max() - pairs[ft].min())
            nf2 = feature + '_' + ft + '_2'
            pairs[nf2] = 1.0 * pairs[ft] / pairs[ft_2]
            pairs[nf2] = pairs[nf2].apply(fun)
            old = pd.concat([old, pairs[nf1]], axis=1)
            old = pd.concat([old, pairs[nf2]], axis=1)
        del(pairs)

    pairs = old[['clue_id', 'clue_id_2']]
    for feature in num_vars:
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        pairs[feature] = 1.0 * pairs[feature] / pairs[feature].max()
        pairs[feature_2] = 1.0 * pairs[feature_2] / pairs[feature_2].max()
    num_vars_2 = [feature + '_2' for feature in num_vars]
    mp = pairs[num_vars].as_matrix()
    mp2 = pairs[num_vars_2].as_matrix()
    a = []
    for i in range(len(mp)):
        a.append([cosine(mp[i], mp2[i]), euclidean(mp[i], mp2[i]), braycurtis(mp[i], mp2[i]), canberra(mp[i], mp2[i]), chebyshev(mp[i], mp2[i]), cityblock(mp[i], mp2[i]), correlation(mp[i], mp2[i]), sqeuclidean(mp[i], mp2[i])])
    a = pd.DataFrame(a, columns = ['distance1', 'ditance2', 'distance3', 'distance4', 'distance5', 'ditance6', 'distance7', 'distance8'])
    old = pd.concat([old, a], axis = 1)

    print old

    old.to_csv('t.tsv', index = False, sep = '\t')

