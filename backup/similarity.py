#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, Text8Corpus
from sklearn.metrics import roc_auc_score
import exceptions


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

pairs['similarity'] = 0
count = 0
for i in range(len(pairs)):
    if count % 1000 == 0:
        print str(count) + '/' + str(len(pairs))
    count += 1
    simi = 0.0
    for feature in features:
        feature_2 = feature + '_2'
        if pairs.iloc[i][feature] not in model1.index2word or pairs.iloc[i][feature_2] not in model1.index2word:
            simi += 0
        else:
            simi += model1.similarity(pairs.iloc[i][feature], pairs.iloc[i][feature_2])
    simi = 1.0 * simi / (2 * len(features))
    pairs.iloc[i]['similarity'] = simi

au = roc_auc_score(pairs.LABEL, pairs.similarity)
print au
