#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, Text8Corpus
from sklearn.metrics import roc_auc_score

inp = '../../data/word2vec/user_doc.txt'
outp1 = '../../data/word2vec/user_doc.text.model'
outp2 = '../../data/word2vec/user_doc.text.vector'

model1 = Word2Vec.load(outp1)
print model1.similarity('tag_id17387', 'tag_id2864')

features = ['city_id', 'source_level', 'road_haul', 'transfer_num', 'guobie', 'minor_category_id', 'tag_id', 'car_year', 'auto_type', 'carriages', 'seats', 'fuel_type', 'gearbox', 'air_displacement', 'emission_standard', 'car_color', 'clue_source_type', 'plate_city_id', 'evaluate_score']
test_file = '../../data/testData/dataset.tsv'
car_file = '../../data/hl_car.tsv'
car = pd.read_csv(car_file, sep = '\t')
test = pd.read_csv(test_file, sep = '\t')
pairs = test[['LABEL', 'clue_id', 'clue_id_2']]

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
    pairs['similarity'] += pairs_temp.apply(simi, axis=1)
    print pairs.similarity
    del(pairs_temp)
pairs['similarity'] = 1.0 * pairs['similarity'] / (2 * len(features))

au = roc_auc_score(pairs.LABEL, pairs.similarity)
print au
