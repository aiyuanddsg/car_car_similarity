#!/usr/bin/env python
# encoding: utf-8


import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor

test = pd.read_csv("../../data/testData/test4.tsv", sep='\t')
test = test.fillna(0)
car = pd.read_csv("../../data/hl_car.tsv", sep = '\t')

cat_vars = ['plate_city_id', 'city_id', 'guobie', 'minor_category_id', 'tag_id', 'auto_type', 'fuel_type', 'car_color', 'emission_standard', 'car_year_t', 'gearbox_t', 'carriages_t', 'seats_t']
cat_vars1 = ['plate_city_id', 'city_id', 'guobie', 'minor_category_id', 'tag_id', 'auto_type', 'fuel_type', 'car_color', 'emission_standard']
num_vars = ['suggest_price', 'base_price', 'price', 'road_haul', 'car_year', 'air_displacement', 'gearbox', 'carriages', 'seats']
time_vars = ['license', 'business', 'strong', 'audit']
num_vars3 = ['suggest_price_3', 'base_price_3', 'price_3', 'road_haul_3', 'car_year_3', 'air_displacement_3', 'gearbox_3', 'carriages_3', 'seats_3']

#sft = cat_vars + num_vars + num_vars1 + num_vars2 + num_vars3 + time_vars + num_vars4

sft = list(test.columns)
sft = sft[3:]
sft.remove('car_id')

'''
for var in time_vars:
    test[var].replace(0,1, inplace = True)
    test[var] = 1 / (test[var] + 1)

for var in num_vars:
    test[var].replace(0, test[var].mean() / 2, inplace = True)

for var in num_vars3:
    test[var] = 1.0 * test[var] / test[var].max()
'''

X_test=test[sft]
y_test=test.LABEL
skf = StratifiedKFold(y_test, n_folds=2, shuffle=True)

gbdt = GradientBoostingRegressor(loss='ls'
    , learning_rate=0.1
    , n_estimators=150
    , subsample=1
    , min_samples_split=2
    , min_samples_leaf=2
    , max_depth=10
    , init=None
    , random_state=None
    , max_features=None
    , verbose=0
    , max_leaf_nodes=None
    , warm_start=False
)

for tr, te in skf:
    gbdt = gbdt.fit(X_test.ix[tr], y_test.ix[tr])
    pred = gbdt.predict(X_test.ix[te])
    au = roc_auc_score(y_test.ix[te], pred)
    print au
    fi = pd.Series(gbdt.feature_importances_, index = sft)
    print fi.sort_values(ascending = False)
