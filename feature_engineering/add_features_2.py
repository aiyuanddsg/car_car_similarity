import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
from scipy.spatial.distance import *
import math
from scipy.special import rel_entr, entr
from scipy.stats import entropy
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, Text8Corpus
from sklearn.metrics import roc_auc_score

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

if __name__ =="__main__":
    old = pd.read_csv('../../data/testData/test1.tsv', sep = '\t')
    car = pd.read_csv('../../data/hl_car.tsv', sep = '\t')
    #num_vars = ['road_haul', 'air_displacement']
    num_vars = ['suggest_price', 'base_price', 'price', 'road_haul', 'air_displacement']
    num_vars1 = ['suggest_price', 'base_price', 'price', 'road_haul', 'car_year', 'air_displacement', 'gearbox', 'carriages', 'seats']
    cat_vars = ['plate_city_id', 'city_id', 'guobie', 'minor_category_id', 'tag_id', 'auto_type', 'fuel_type', 'car_color', 'emission_standard', 'car_year', 'gearbox', 'carriages', 'seats']
    #t = add_feature(old, car, num_vars)
    fts = ['clue_id', 'clue_id_2', 'LABEL', 'plate_city_id', 'city_id', 'guobie', 'minor_category_id', 'tag_id', 'car_id', 'auto_type', 'fuel_type', 'car_color', 'emission_standard', 'car_year_t', 'gearbox_t', 'carriages_t', 'seats_t', 'road_haul', 'car_year', 'air_displacement', 'gearbox', 'carriages', 'seats', 'road_haul_1', 'car_year_1', 'air_displacement_1', 'gearbox_1', 'carriages_1', 'seats_1', 'road_haul_2', u'car_year_2', 'air_displacement_2', 'gearbox_2', 'carriages_2', 'seats_2', 'road_haul_3', 'car_year_3', 'air_displacement_3', 'gearbox_3', 'carriages_3', 'seats_3', 'license', 'business', 'strong', 'audit']
    old = old[fts]

    inp = '../../data/word2vec/user_doc.txt'
    outp1 = '../../data/word2vec/user_doc.text.model'
    outp2 = '../../data/word2vec/user_doc.text.vector'

    model1 = Word2Vec.load(outp1)
    #print model1.similarity('tag_id17387', 'tag_id2864')

    features = ['city_id', 'source_level', 'road_haul', 'transfer_num', 'guobie', 'minor_category_id', 'tag_id', 'car_year', 'auto_type', 'carriages', 'seats', 'fuel_type', 'gearbox', 'air_displacement', 'emission_standard', 'car_color', 'clue_source_type', 'plate_city_id', 'evaluate_score']
    pairs = old[['LABEL', 'clue_id', 'clue_id_2']]


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

    '''
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
    '''

    dis = 'SO_distance'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs[dis] = 0
    for feature in cat_vars:
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        new_feature = feature + '_SO'
        feature_2 = feature + '_2'
        new_feature_2 = feature + 'SO_2'
        Sx = 100 / (car[feature].max() - car[feature].min())
        Ox = -car[feature].min()
        pairs[new_feature] = pairs[feature] * Sx + Ox
        pairs[new_feature_2] = pairs[feature_2] * Sx + Ox
        #old = pd.concat([old, pairs[new_feature]], axis=1)
        #old = pd.concat([old, pairs[new_feature_2]], axis=1)
        pairs[dis] += (pairs[new_feature] - pairs[new_feature_2]) ** 2
    old = pd.concat([old, pairs[dis]], axis=1)
    print old[dis]
    del(pairs)
    old[dis] = old[dis].apply(math.sqrt)
    old[dis] = (old[dis] - old[dis].mean()) / old[dis].std()
    print old[dis]

    dis = 'KL_distance'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs[dis] = 0
    for feature in cat_vars:
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        new_feature = feature + '_KL'
        feature_2 = feature + '_2'
        new_feature_2 = feature + 'KL_2'
        sm = car[feature].unique().sum()
        pairs[feature] = 1.0 * pairs[feature] / sm
        pairs[feature_2] = 1.0 * pairs[feature_2] / sm
        pairs[dis] += rel_entr(pairs[feature], pairs[feature_2]) + rel_entr(pairs[feature_2], pairs[feature])
    old = pd.concat([old, pairs[dis]], axis=1)
    del(pairs)
    old[dis].replace([np.inf, -np.inf], 0, inplace = True)
    old[dis] = (old[dis] - old[dis].mean()) / old[dis].std()
    #print old[dis]

    dis = 'Overlap'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs[dis] = 0
    w = 1.0 / len(cat_vars)
    for feature in cat_vars:
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        pairs['temp'] = abs(pairs[feature] - pairs[feature_2]) > 0
        pairs['temp'] = pairs['temp'].astype(str).replace({'True': 0, 'False': 1})
        pairs[dis] += w * pairs['temp']
    old = pd.concat([old, pairs[dis]], axis=1)
    del(pairs)
    old[dis].replace([np.inf, -np.inf], 0, inplace = True)
    old[dis] = (old[dis] - old[dis].mean()) / old[dis].std()
    print old[dis]

    dis = 'Eskin'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs[dis] = 0
    w = 1.0 / len(cat_vars)
    for feature in cat_vars:
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        pairs['temp'] = abs(pairs[feature] - pairs[feature_2]) > 0
        nk = pairs[feature].nunique()
        nk2 = nk ** 2
        nkk = 1.0 * nk2 / (nk2 + 2)
        pairs['temp'] = pairs['temp'].astype(str).replace({'True': 1, 'False': 0})
        pairs[dis] += w * pairs['temp']
    old = pd.concat([old, pairs[dis]], axis=1)
    del(pairs)
    old[dis].replace([np.inf, -np.inf], 0, inplace = True)
    old[dis] = (old[dis] - old[dis].mean()) / old[dis].std()
    print old[dis]

    dis = 'IOF'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs[dis] = 0
    w = 1.0 / len(cat_vars)
    for feature in cat_vars:
        print feature
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        pairs['temp'] = abs(pairs[feature] - pairs[feature_2]) > 0
        pairs['temp'] = pairs['temp'].astype(str).replace({'True': 1, 'False': 0})
        unq = pairs[feature].unique()
        nkdict = {}
        for u in unq:
            nkdict[u] = len(pairs[pairs[feature] == u])
        unq_2 = pairs[feature_2].unique()
        nkdict_2 = {}
        for u_2 in unq_2:
            nkdict_2[u_2] = len(pairs[pairs[feature_2] == u_2])
        new_feature = feature + '_IOF'
        new_feature_2 = feature + '_IOF_2'
        pairs[new_feature] = pairs[feature].replace(nkdict)
        pairs[new_feature] = pairs[new_feature].apply(math.log)
        pairs[new_feature_2] = pairs[feature_2].replace(nkdict_2)
        pairs[new_feature_2] = pairs[new_feature_2].apply(math.log)
        pairs['temp'] = pairs['temp'] * 1.0 / (1 + pairs[new_feature] * pairs[new_feature_2])
        pairs['temp'] = pairs['temp'].replace({0: 1})
        pairs[dis] += w * pairs['temp']
    old = pd.concat([old, pairs[dis]], axis=1)
    del(pairs)
    old[dis].replace([np.inf, -np.inf], 0, inplace = True)
    old[dis] = (old[dis] - old[dis].mean()) / old[dis].std()
    print old[dis]

    dis = 'OF'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs[dis] = 0
    N = len(pairs)
    w = 1.0 / len(cat_vars)
    for feature in cat_vars:
        print feature
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        pairs['temp'] = abs(pairs[feature] - pairs[feature_2]) > 0
        pairs['temp'] = pairs['temp'].astype(str).replace({'True': 1, 'False': 0})
        unq = pairs[feature].unique()
        nkdict = {}
        for u in unq:
            nkdict[u] = len(pairs[pairs[feature] == u])
        unq_2 = pairs[feature_2].unique()
        nkdict_2 = {}
        for u_2 in unq_2:
            nkdict_2[u_2] = len(pairs[pairs[feature_2] == u_2])
        new_feature = feature + '_OF'
        new_feature_2 = feature + '_OF_2'
        pairs[new_feature] = pairs[feature].replace(nkdict)
        pairs[new_feature] = 1.0 * N / pairs[new_feature]
        pairs[new_feature] = pairs[new_feature].apply(math.log)
        pairs[new_feature_2] = pairs[feature_2].replace(nkdict_2)
        pairs[new_feature_2] = 1.0 * N / pairs[new_feature_2]
        pairs[new_feature_2] = pairs[new_feature_2].apply(math.log)
        pairs['temp'] = pairs['temp'] * 1.0 / (1 + pairs[new_feature] * pairs[new_feature_2])
        pairs['temp'] = pairs['temp'].replace({0: 1})
        pairs[dis] += w * pairs['temp']
    old = pd.concat([old, pairs[dis]], axis=1)
    del(pairs)
    old[dis].replace([np.inf, -np.inf], 0, inplace = True)
    old[dis] = (old[dis] - old[dis].mean()) / old[dis].std()
    print old[dis]

    dis = 'Lin'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs[dis] = 0
    N = len(pairs)
    for feature in cat_vars:
        print feature
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        pairs['temp'] = abs(pairs[feature] - pairs[feature_2]) > 0
        pairs['temp_1'] = pairs['temp']
        pairs['temp'] = pairs['temp'].astype(str).replace({'True': 0, 'False': 1})
        pairs['temp_1'] = pairs['temp'].astype(str).replace({'True': 1, 'False': 0})
        unq = pairs[feature].unique()
        nkdict = {}
        for u in unq:
            nkdict[u] = len(pairs[pairs[feature] == u])
        unq_2 = pairs[feature_2].unique()
        nkdict_2 = {}
        for u_2 in unq_2:
            nkdict_2[u_2] = len(pairs[pairs[feature_2] == u_2])
        new_feature = feature + '_Lin'
        new_feature_2 = feature + '_Lin_2'
        pairs[new_feature] = pairs[feature].replace(nkdict)
        pairs[new_feature] = 1.0 * pairs[new_feature] / N
        pairs[new_feature_2] = pairs[feature_2].replace(nkdict_2)
        pairs[new_feature_2] = 1.0 * pairs[new_feature_2] / N
        pairs['sum'] = pairs[new_feature] + pairs[new_feature_2]
        pairs[new_feature] = pairs[new_feature].apply(math.log)
        pairs[new_feature_2] = pairs[new_feature_2].apply(math.log)
        pairs['sum'] = pairs['sum'].apply(math.log)
        pairs['temp'] = pairs['temp'] * 2.0 * pairs[new_feature]
        print pairs['sum']
        pairs['temp_1'] = pairs['temp_1'].astype(float) * 2.0 * pairs['sum']
        sm = pairs[new_feature].unique().sum() + pairs[new_feature_2].unique().sum()
        w = 1.0 / sm
        pairs[dis] = pairs[dis] + w * (pairs['temp'] + pairs['temp_1'])
    old = pd.concat([old, pairs[dis]], axis=1)
    del(pairs)
    old[dis].replace([np.inf, -np.inf], 0, inplace = True)
    old[dis] = (old[dis] - old[dis].mean()) / old[dis].std()
    print old[dis]


    dis = 'Goodall1'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs[dis] = 0
    N = len(pairs)
    w = 1.0 / len(cat_vars)
    for feature in cat_vars:
        print feature
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        pairs['temp'] = abs(pairs[feature] - pairs[feature_2]) > 0
        pairs['temp'] = pairs['temp'].astype(str).replace({'True': 0, 'False': 1})
        unq = pairs[feature].unique()
        unq_2 = pairs[feature_2].unique()
        all_unq = set(unq) | set(unq_2)
        nkdict = {}
        for u in all_unq:
            nkdict[u] = 0
            if u in unq:
                nkdict[u] += len(pairs[pairs[feature] == u])
            if u in unq_2:
                nkdict[u] += len(pairs[pairs[feature_2] == u])
            nkdict[u] = 1.0 * nkdict[u] / (2 * N)
        sum_dict = {}
        for i in nkdict.keys():
            sum_dict[nkdict[i]] = 0
            for j in nkdict.keys():
                if nkdict[j] < nkdict[i]:
                    sum_dict[nkdict[i]] += nkdict[j] ** 2
            sum_dict[nkdict[i]] = 1 - sum_dict[nkdict[i]]
        new_feature = feature + '_Doodall1'
        pairs[new_feature] = pairs[feature].replace(nkdict).replace(sum_dict)
        pairs['temp'] = pairs['temp'] * 1.0 * pairs[new_feature]
        pairs[dis] = pairs[dis] + w * pairs['temp']
    old = pd.concat([old, pairs[dis]], axis=1)
    del(pairs)
    old[dis].replace([np.inf, -np.inf], 0, inplace = True)
    old[dis] = (old[dis] - old[dis].mean()) / old[dis].std()
    print old[dis]


    dis = 'Goodall2'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs[dis] = 0
    N = len(pairs)
    w = 1.0 / len(cat_vars)
    for feature in cat_vars:
        print feature
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        pairs['temp'] = abs(pairs[feature] - pairs[feature_2]) > 0
        pairs['temp'] = pairs['temp'].astype(str).replace({'True': 0, 'False': 1})
        unq = pairs[feature].unique()
        unq_2 = pairs[feature_2].unique()
        all_unq = set(unq) | set(unq_2)
        nkdict = {}
        for u in all_unq:
            nkdict[u] = 0
            if u in unq:
                nkdict[u] += len(pairs[pairs[feature] == u])
            if u in unq_2:
                nkdict[u] += len(pairs[pairs[feature_2] == u])
            nkdict[u] = 1.0 * nkdict[u] / (2 * N)
        sum_dict = {}
        for i in nkdict.keys():
            sum_dict[nkdict[i]] = 0
            for j in nkdict.keys():
                if nkdict[j] > nkdict[i]:
                    sum_dict[nkdict[i]] += nkdict[j] ** 2
            sum_dict[nkdict[i]] = 1 - sum_dict[nkdict[i]]
        new_feature = feature + '_Doodall2'
        pairs[new_feature] = pairs[feature].replace(nkdict).replace(sum_dict)
        pairs['temp'] = pairs['temp'] * 1.0 * pairs[new_feature]
        pairs[dis] = pairs[dis] + w * pairs['temp']
    old = pd.concat([old, pairs[dis]], axis=1)
    del(pairs)
    old[dis].replace([np.inf, -np.inf], 0, inplace = True)
    old[dis] = (old[dis] - old[dis].mean()) / old[dis].std()
    print old[dis]


    dis = 'Goodall3'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs[dis] = 0
    N = len(pairs)
    w = 1.0 / len(cat_vars)
    for feature in cat_vars:
        print feature
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        pairs['temp'] = abs(pairs[feature] - pairs[feature_2]) > 0
        pairs['temp'] = pairs['temp'].astype(str).replace({'True': 0, 'False': 1})
        unq = pairs[feature].unique()
        unq_2 = pairs[feature_2].unique()
        all_unq = set(unq) | set(unq_2)
        nkdict = {}
        for u in all_unq:
            nkdict[u] = 0
            if u in unq:
                nkdict[u] += len(pairs[pairs[feature] == u])
            if u in unq_2:
                nkdict[u] += len(pairs[pairs[feature_2] == u])
            nkdict[u] = 1.0 * nkdict[u] / (2 * N)
        sum_dict = {}
        for i in nkdict.keys():
            sum_dict[nkdict[i]] = 1 - nkdict[i] ** 2
        new_feature = feature + '_Goodall3'
        pairs[new_feature] = pairs[feature].replace(nkdict).replace(sum_dict)
        pairs['temp'] = pairs['temp'] * 1.0 * pairs[new_feature]
        pairs[dis] = pairs[dis] + w * pairs['temp']
    old = pd.concat([old, pairs[dis]], axis=1)
    del(pairs)
    old[dis].replace([np.inf, -np.inf], 0, inplace = True)
    old[dis] = (old[dis] - old[dis].mean()) / old[dis].std()
    print old[dis].describe()


    dis = 'Goodall4'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs[dis] = 0
    N = len(pairs)
    w = 1.0 / len(cat_vars)
    for feature in cat_vars:
        print feature
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        pairs['temp'] = abs(pairs[feature] - pairs[feature_2]) > 0
        pairs['temp'] = pairs['temp'].astype(str).replace({'True': 0, 'False': 1})
        unq = pairs[feature].unique()
        unq_2 = pairs[feature_2].unique()
        all_unq = set(unq) | set(unq_2)
        nkdict = {}
        for u in all_unq:
            nkdict[u] = 0
            if u in unq:
                nkdict[u] += len(pairs[pairs[feature] == u])
            if u in unq_2:
                nkdict[u] += len(pairs[pairs[feature_2] == u])
            nkdict[u] = 1.0 * nkdict[u] / (2 * N)
        sum_dict = {}
        for i in nkdict.keys():
            sum_dict[nkdict[i]] = nkdict[i] ** 2
        new_feature = feature + '_Goodall2'
        pairs[new_feature] = pairs[feature].replace(nkdict).replace(sum_dict)
        pairs['temp'] = pairs['temp'] * 1.0 * pairs[new_feature]
        pairs[dis] = pairs[dis] + w * pairs['temp']
    old = pd.concat([old, pairs[dis]], axis=1)
    del(pairs)
    old[dis].replace([np.inf, -np.inf], 0, inplace = True)
    old[dis] = (old[dis] - old[dis].mean()) / old[dis].std()
    print old[dis].describe()

    dis = 'Gambaryan'
    pairs = old[['clue_id', 'clue_id_2']]
    pairs[dis] = 0
    N = len(pairs)
    w = 1.0 / len(cat_vars)
    for feature in cat_vars:
        print feature
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id')
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2')
        feature_2 = feature + '_2'
        pairs['temp'] = abs(pairs[feature] - pairs[feature_2]) > 0
        pairs['temp'] = pairs['temp'].astype(str).replace({'True': 0, 'False': 1})
        unq = pairs[feature].unique()
        unq_2 = pairs[feature_2].unique()
        all_unq = set(unq) | set(unq_2)
        nkdict = {}
        for u in all_unq:
            nkdict[u] = 0
            if u in unq:
                nkdict[u] += len(pairs[pairs[feature] == u])
            if u in unq_2:
                nkdict[u] += len(pairs[pairs[feature_2] == u])
            nkdict[u] = 1.0 * nkdict[u] / (2 * N)
        sum_dict = {}
        for i in nkdict.keys():
            sum_dict[nkdict[i]] = -(nkdict[i] * math.log(nkdict[i]) + (1 - nkdict[i]) * math.log(1 - nkdict[i]))
        new_feature = feature + '_Gambaryan'
        pairs[new_feature] = pairs[feature].replace(nkdict).replace(sum_dict)
        pairs['temp'] = pairs['temp'] * 1.0 * pairs[new_feature]
        pairs[dis] = pairs[dis] + w * pairs['temp']
    old = pd.concat([old, pairs[dis]], axis=1)
    del(pairs)
    old[dis].replace([np.inf, -np.inf], 0, inplace = True)
    old[dis] = (old[dis] - old[dis].mean()) / old[dis].std()
    print old[dis].describe()

    pairs = old[['clue_id', 'clue_id_2']]
    for feature in features:
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id').dropna()
        pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_2').dropna()

    features_2 = [feature + '_2' for feature in features]
    all_features = features + features_2
    pairs[all_features] = pairs[all_features].astype(int).astype(str)

    for feature in features:
        feature_2 = feature + '_2'
        pairs[feature] = feature + pairs[feature]
        pairs[feature_2] = feature + pairs[feature_2]

    def simi(x):
        if x[0] not in model1.index2word or x[1] not in model1.index2word:
            return 0
        return model1.similarity(x[0], x[1])

    pairs['similarity'] = 0
    for feature in features:
        feature_2 = feature + '_2'
        pairs_temp = pairs[[feature, feature_2]]
        pairs_temp.columns = [0,1]
        new_feature = 'word2vec_' + feature
        pairs[new_feature] = pairs_temp.apply(simi, axis=1)
        old = pd.concat([old, pairs[new_feature]], axis=1)
        del(pairs_temp)
    del(pairs)


    print old
    old.to_csv('../../data/testData/test4.tsv', index = False, sep = '\t')

