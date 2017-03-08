import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
from no_dealer_transformation_probability import *
from generate_features import *
from add_features import *


if __name__ =="__main__":

    cat_vars = ['plate_city_id', 'city_id', 'guobie', 'minor_category_id', 'tag_id', 'car_id', 'auto_type', 'fuel_type', 'car_color', 'emission_standard', 'car_year', 'gearbox', 'carriages', 'seats']
    num_vars = ['suggest_price', 'base_price', 'price', 'road_haul', 'air_displacement', 'car_year', 'gearbox', 'carriages', 'seats']
    time_vars = [['license_date', 'license_month'], ['business_insurance_year', 'business_insurance_month'], ['strong_insurance_year', 'strong_insurance_month'], ['audit_year', 'audit_month']]

    car_file = '../../data/hl_car.tsv'
    appoint_file = '../../data/hl_appoint.tsv'
    test_file = '../../data/testData/dataset.tsv'

    car = pd.read_csv(car_file, sep = '\t')
    test = pd.read_csv(test_file, sep='\t')

    test_pairs = test[['LABEL', 'clue_id', 'clue_id_2']]

    print 'no dealer'
    ndtp = no_dealer_transformation_probability(appoint_file, car_file, cat_vars)
    nd_pairs, nd_tp = ndtp.run()

    #test_pairs = test_pairs.sample(n = 5000, replace=True)

    print 'generate trans'
    gf = generate_features(car_file, test_pairs, nd_tp, cat_vars, num_vars, time_vars)
    inpt = gf.run()

    '''
    cat_vars_1 = cat_vars.remove('car_id')
    af = add_features(inpt, car_file, num_vars, cat_vars)
    all_features = af.run()
    all_features.to_csv('../../data/testData/all_features.tsv', index = False, sep = '\t')
    '''
