import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

#train = pd.read_csv("../../data/categoryLR_input/all_features_train.tsv", sep = '\t')
#train['LABEL']=train['LABEL'].astype(int)
test = pd.read_csv("../../data/testData/test1.tsv", sep='\t')
car = pd.read_csv("../../data/hl_car.tsv", sep='\t')
'''
features = ['emission_standard', 'car_year', 'transfer_num', 'minor_category_id', 'car_id', 'tag_id', 'source_level', 'auto_type', 'carriages', 'fuel_type', 'gearbox', 'car_color', 'guobie', 'seats']
numerical_features = ['price']
'''
cat_vars = ['city_id', 'guobie', 'minor_category_id', 'tag_id', 'auto_type', 'fuel_type', 'car_color', 'emission_standard', 'car_year_2', 'gearbox_2', 'carriages_2', 'seats_2']
#cat_vars = ['city_id', 'guobie', 'minor_category_id', 'tag_id', 'car_id', 'auto_type', 'fuel_type', 'car_color', 'emission_standard', 'car_year_2', 'gearbox_2', 'carriages_2', 'seats_2']
print len(cat_vars)
num_vars = ['price', 'road_haul', 'car_year', 'air_displacement', 'gearbox', 'carriages', 'seats']
#num_vars = ['price', 'road_haul', 'air_displacement']
num_vars1 = ['price_1', 'road_haul_1', 'car_year_1', 'air_displacement_1', 'gearbox_1', 'carriages_1', 'seats_1']
time_vars = ['license', 'business', 'strong', 'audit']

sft = cat_vars + num_vars + num_vars1

for var in time_vars:
    test[var].replace(0,1, inplace = True)
    test[var] = 1 / (test[var] + 1)

for var in num_vars:
    test[var].replace(0, test[var].mean() / 2, inplace = True)

X_test=test[sft]
y_test=test.LABEL

skf = StratifiedKFold(y_test, n_folds=2, shuffle=True)
clf = LogisticRegression(class_weight={1:10, 0:1})
for tr, te in skf:
    clf = clf.fit(X_test.ix[tr], y_test.ix[tr])
    co=clf.coef_
    co=co[0]
    co=list(co)
    coe=pd.Series(co, index=sft)
    coe = coe.reset_index([i for i in range(len(coe))])
    print coe
    su=co[0] * X_test.ix[te][sft[0]]
    for i in range(1, len(co)):
        su+=co[i]*X_test.ix[te][sft[i]]

    pairs = test.ix[te][['clue_id', 'clue_id_2']]
    pairs = pairs.join(car.set_index('clue_id')['car_id'], on = 'clue_id')
    pairs = pairs.join(car.set_index('clue_id')['car_id'], on = 'clue_id_2', rsuffix = '_2')
    temp = pairs.car_id == pairs.car_id_2
    for i in range(len(su)):
        if temp.iloc[i] == True:
            su.iloc[i] = 1000
    print su


    au = roc_auc_score(y_test.ix[te], su)
    print au

