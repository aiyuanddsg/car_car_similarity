import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
import random as rd

def set_feature():
    feature = ['emission_standard', 'car_year', 'transfer_num', 'minor_category_id', 'source_level', 'auto_type', 'carriages', 'fuel_type', 'gearbox', 'car_color', 'guobie', 'seats']
    return feature

def get_file_name(feature, isDealer):
    if isDealer == True:
        return '../../data/categoryTransFormationProbability/' + feature + 'NoDealer.tsv'
    else:
        return '../../data/categoryTransFormationProbability/' + feature + 'NoDealerNegativeSample.tsv'

def read_file(filename):
    fileContent = pd.read_csv(filename, sep = '\t')
    return fileContent

def get_no_dealer_appoint(appoint):
    appoint = appoint[appoint['dealer'] == 0]
    return appoint

def generate_clue_pairs(appoint):
    groups = appoint.sort_values('create_time').groupby('user_id', sort = False)
    pairs = []
    for idx, g in groups:
        clues = list(g['clue_id'])
        for i in range(len(clues) - 1):
            j = i + 1
            pairs.append([clues[i], clues[j]])
    return pairs

def get_user_pairs(appoint, car):
    ap = appoint.join(car.set_index('clue_id')['price'], on = 'clue_id')
    ap = ap[['user_id', 'price']]
    mean_price = ap.groupby('user_id').mean()
    mean_price = mean_price[mean_price['price'] > 0]
    amp = mean_price.sort_values('price', ascending=True)
    idx = amp.index
    user_pairs = []
    count = 0
    for i in range(0, 1000):
        j = rd.randint(0, len(idx) / 2)
        k = rd.randint(len(idx) / 2 + 1, len(idx) - 1)
        if amp.ix[idx[j]]['price'] * 4 < amp.ix[idx[k]]['price']:
            if rd.random() < 0.5:
                user_pairs.append([idx[j], idx[k]])
            else:
                user_pairs.append([idx[k], idx[j]])
    return user_pairs

def get_clue_pairs(user_pairs, appoint):
    ap = appoint[['user_id', 'clue_id']]
    ap = ap.groupby('user_id').groups
    clue_pairs = []
    for pair in user_pairs:
        for clue1 in ap[pair[0]]:
            for clue2 in ap[pair[1]]:
                clue_pairs.append([clue1, clue2])
   # clue_pairs = pd.DataFrame(clue_pairs, columns = ['clue_id', 'clue_id_2'])
    return clue_pairs

def generate_input(pairs, features, transformation_probability, car):
    tempCar = car.set_index('clue_id')
    similaritys = []
    count = 0
    for pair in pairs:
        if count == 0:
            print type(tempCar.index)
            print type(pair)
        count = count + 1
   #     print str(count) + ' / ' + str(len(pairs))
        if pair[0] not in tempCar.index or pair[1] not in tempCar.index:
     #       print 1
            continue
        similarity = []
        for feature in features:
            clue_feature_value = tempCar.ix[pair[0]][feature]
            the_other_clue_feature_value = tempCar.ix[pair[1]][feature]

            if clue_feature_value <= 0 or the_other_clue_feature_value <= 0:
 #               print 2
                continue

            idx = list(transformation_probability[feature].set_index(feature).index)
            idx_int = [int(id) for id in idx]
            if int(clue_feature_value) in idx_int:
                clue_feature_value = idx[idx_int.index(int(clue_feature_value))]
            else:
  #              print 3
                continue

            idx = list(transformation_probability[feature].set_index(feature).columns)
            if '.' in idx[0]:
                idx_int = [int(float(id)) for id in idx]
            else:
                idx_int = [int(id) for id in idx]
            if int(the_other_clue_feature_value) in idx_int:
                the_other_clue_feature_value = idx[idx_int.index(int(the_other_clue_feature_value))]
            else:
   #             print 4
                continue

            similarity.append(transformation_probability[feature].set_index(feature).ix[clue_feature_value][the_other_clue_feature_value])
        if len(similarity) > 0:
            similaritys.append(similarity)
    return similaritys

def get_input_file_name():
    return '../../data/categoryLR_input/input.csv'

def write_input(file_name, no_dealer_input, all_input):
    f = open(file_name, "w")
    for i in range(len(no_dealer_input)):
        f.write(str(1))
        count = 0
        for j in range(0, len(no_dealer_input[i])):
            f.write(' ' + str(count) + ':' + str(no_dealer_input[i][j]))
            count = count + 1
        f.write('\n')
    for i in range(len(all_input)):
        f.write(str(-1))
        count = 0
        for j in range(0, len(all_input[i])):
            f.write(' ' + str(count) + ':' + str(all_input[i][j]))
            count = count + 1
        f.write('\n')
    f.close()


if __name__ =="__main__":
    appoint = read_file('../../data/hl_appoint.tsv')
    car = read_file('../../data/hl_car.tsv')

    features = set_feature()
    '''
    no_dealer_appoint = get_no_dealer_appoint(appoint)
    no_dealer_pairs = generate_clue_pairs(no_dealer_appoint)
    '''
    user_pairs = get_user_pairs(appoint, car)
    pairs = get_clue_pairs(user_pairs, appoint)
    #no_dealer_transformation_probability = {}
    transformation_probability = {}
    for feature in features:
        print feature
     #   no_dealer_feature_name = get_file_name(feature, True)
        feature_name = get_file_name(feature, False)
      #  no_dealer_transformation_probability[feature] = read_file(no_dealer_feature_name)
        transformation_probability[feature] = read_file(feature_name)

    #no_dealer_input = generate_input(no_dealer_pairs, features, no_dealer_transformation_probability, car)

    all_input = generate_input(pairs, features, transformation_probability, car)

    input_file_name = get_input_file_name()
    #write_input(input_file_name, no_dealer_input, all_input)
