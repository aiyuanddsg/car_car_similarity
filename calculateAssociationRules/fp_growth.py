import pandas as pd
import pyfpgrowth

car = pd.read_csv('../../data/hl_car.tsv', sep = '\t')
appoint = pd.read_csv('../../data/hl_appoint.tsv', sep = '\t')
appoint = appoint[appoint['dealer'] == 0]
new_appoint = appoint.join(car.set_index('clue_id'), on = 'clue_id', how = 'left', rsuffix = '_2')
features = ['emission_standard', 'car_year', 'transfer_num', 'minor_category_id', 'car_id', 'tag_id', 'source_level', 'auto_type', 'carriages', 'fuel_type', 'gearbox', 'car_color', 'guobie', 'seats']

#new_appoint = new_appoint.iloc[:10000]

new_appoint = new_appoint.sort_values('create_time')
groups = new_appoint.groupby('user_id')
pts = {}
for feature in features:
    print feature
    fp = []
    for idx, g in groups:
        temp = g[g[feature] > 0]
        fp.append(list(temp[feature]))
    patterns = pyfpgrowth.find_frequent_patterns(fp, 50)
    pts[feature] = patterns
    '''
    rules = pyfpgrowth.generate_association_rules(patterns, 0)
    if feature == 'gearbox':
        print patterns
        print type(patterns.keys()[0])
    '''
