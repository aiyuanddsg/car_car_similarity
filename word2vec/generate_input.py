#!/usr/bin/env python
# encoding: utf-8

import pandas as pd


class User_doc:
    def __init__(self, appoint_file_name, car_file_name, features, user_doc_file_name):
        self.appoint_file_name = appoint_file_name
        self.car_file_name = car_file_name
        self.features = features
        self.user_doc_file_name = user_doc_file_name

    def readFile(self, file_name):
        fileContent = pd.read_csv(file_name, sep = '\t')
        return fileContent

    def appoint_no_dealer(self, appoint):
        appoint = appoint[appoint['dealer'] == 0]
        return appoint

    def new_appoint(self, appoint):
        columns = ['clue_id', 'user_id']
        temp_appoint = appoint[columns].dropna()
        return temp_appoint

    def generate_user_doc(self, appoint, car):
        for feature in self.features:
            appoint = appoint.join(car.set_index('clue_id')[feature], on = 'clue_id').dropna()
        appoint[self.features] = appoint[self.features].astype(int).astype(str)
        for feature in self.features:
            appoint[feature] = feature + appoint[feature]

        groups = appoint.groupby('user_id', sort = False)
        wfile = open(self.user_doc_file_name, 'w')
        for idx, g in groups:
            for feature in self.features:
                ls = list(g[feature])
                for word in ls:
                    wfile.write(word + ' ')
            wfile.write('\n')

    def run(self):
        car = self.readFile(self.car_file_name)
        appoint = self.readFile(self.appoint_file_name)
        apt_no_dl = self.appoint_no_dealer(appoint)
        na = self.new_appoint(apt_no_dl)
        self.generate_user_doc(na, car)


if __name__ =="__main__":
    car_file = '../../data/hl_car.tsv'
    appoint_file = '../../data/hl_appoint.tsv'
    features = ['city_id', 'source_level', 'road_haul', 'transfer_num', 'guobie', 'minor_category_id', 'tag_id', 'car_year', 'auto_type', 'carriages', 'seats', 'fuel_type', 'gearbox', 'air_displacement', 'emission_standard', 'car_color', 'clue_source_type', 'plate_city_id', 'evaluate_score']
    user_doc_file = '../../data/word2vec/user_doc.txt'
    ud = User_doc(appoint_file, car_file, features, user_doc_file)
    ud.run()



'''
car = pd.read_csv('../../data/hl_car.tsv', sep = '\t')
appoint = pd.read_csv('../../data/hl_appoint.tsv', sep = '\t')
appoint = appoint[appoint.dealer == 0]
doc = appoint[['clue_id']]
features = ['city_id', 'source_level', 'road_haul', 'transfer_num', 'guobie', 'minor_category_id', 'tag_id', 'car_year', 'auto_type', 'carriages', 'seats', 'fuel_type', 'gearbox', 'air_displacement', 'emission_standard', 'car_color', 'clue_source_type', 'plate_city_id', 'evaluate_score']

for feature in features:
    doc = doc.join(car.set_index('clue_id')[feature], on = 'clue_id')

doc = doc.astype(str)
for feature in features:
    doc[feature] = doc[feature] + feature

print len(doc)
print doc.clue_id.nunique()
'''

