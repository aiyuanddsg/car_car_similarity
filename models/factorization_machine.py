import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer

test = pd.read_csv("../../data/testData/test3.tsv", sep='\t')
test = test.fillna(0)
car = pd.read_csv("../../data/hl_car.tsv", sep = '\t')

cat_vars = ['plate_city_id', 'city_id', 'guobie', 'minor_category_id', 'tag_id', 'auto_type', 'fuel_type', 'car_color', 'emission_standard', 'car_year_t', 'gearbox_t', 'carriages_t', 'seats_t']
cat_vars1 = ['plate_city_id', 'city_id', 'guobie', 'minor_category_id', 'tag_id', 'auto_type', 'fuel_type', 'car_color', 'emission_standard']
num_vars = ['suggest_price', 'base_price', 'price', 'road_haul', 'car_year', 'air_displacement', 'gearbox', 'carriages', 'seats']
time_vars = ['license', 'business', 'strong', 'audit']
num_vars3 = ['suggest_price_3', 'base_price_3', 'price_3', 'road_haul_3', 'car_year_3', 'air_displacement_3', 'gearbox_3', 'carriages_3', 'seats_3']

cat_vars_2 = [feature+'_2' for feature in cat_vars1]
num_vars_2 = [feature+'_2' for feature in num_vars]
cat_fts = cat_vars1 + cat_vars_2
num_fts = num_vars + num_vars_2
fts = cat_fts + num_fts

fts = ['LABEL','clue_id', 'clue_id_2'] + cat_vars1 + num_vars
fts1 = cat_vars1 + num_vars
#fts = ['LABEL','clue_id', 'clue_id_2']
pairs = test[fts]

for feature in cat_vars1:
    pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id', rsuffix = '_5')
    pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_6')

for feature in num_vars:
    pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id', rsuffix = '_5')
    pairs = pairs.join(car.set_index('clue_id')[feature], on = 'clue_id_2', rsuffix = '_6')

cat_vars5 = [cv + '_5' for cv in cat_vars1]
cat_vars6 = [cv + '_6' for cv in cat_vars1]
num_vars5 = [num + '_5' for num in num_vars]
num_vars6 = [num + '_6' for num in num_vars]

for num in num_vars5:
    pairs[num] = (pairs[num] - pairs[num].mean()) / pairs[num].std()
for num in num_vars6:
    pairs[num] = (pairs[num] - pairs[num].mean()) / pairs[num].std()

cat_vars56 = cat_vars5 + cat_vars6
num_vars56 = num_vars5 + num_vars6

#pairs[cat_vars12] = pairs[cat_vars12].astype(str)
sft = fts1 + cat_vars56 + num_vars56

print pairs.columns
print pairs

data = pairs[sft].astype(str).to_dict(orient='records')
label = np.array(pairs.LABEL)


v = DictVectorizer()
X = v.fit_transform(data)
print X

fm = pylibfm.FM(num_factors=100, num_iter=150, verbose=True, task="classification", initial_learning_rate=0.001, learning_rate_schedule="optimal")
print "one-hot done!"

y_test=test.LABEL
skf = StratifiedKFold(y_test, n_folds=2, shuffle=True)


for tr, te in skf:
    fm.fit(X[tr], label[tr])
    pred = fm.predict(X[te])
    au = roc_auc_score(label[te], pred)
    print au

