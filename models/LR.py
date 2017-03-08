import pandas as pd
#import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from evaluation import va, eval_features, eval_clf_result
from sklearn.grid_search import GridSearchCV
#import matplotlib.pyplot as plt
from scipy import interp
#from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

#train = pd.read_csv("../../data/train_test/train4.tsv", sep = '\t')
#train['LABEL']=train['LABEL'].astype(int)
test = pd.read_csv("../../data/train_test/test.tsv", sep='\t')
features = ['emission_standard', 'car_year', 'transfer_num', 'minor_category_id', 'car_id', 'tag_id', 'source_level', 'auto_type', 'carriages', 'fuel_type', 'gearbox', 'car_color', 'guobie', 'seats']
numerical_features = ['price']
fts = features + numerical_features

'''
'car_year', 'car_id'
'fuel_type', 'gearbox', 'price'
'''

#sft=['car_id', 'tag_id', 'price', 'gearbox']
sft=['emission_standard', 'car_year', 'transfer_num', 'minor_category_id', 'car_id', 'tag_id', 'source_level', 'auto_type', 'carriages', 'fuel_type', 'gearbox', 'car_color', 'guobie', 'seats', 'price']
'''
X_train=train[sft]
y_train=train.LABEL
'''
X_test=test[sft]
y_test=test.LABEL
skf = StratifiedKFold(y_test, n_folds=5, shuffle=True)
coes={}
clf = LogisticRegression(class_weight={1:10, 0:1})
for tr, te in skf:
    clf = clf.fit(X_test.ix[tr], y_test.ix[tr])
    co=clf.coef_
    co=co[0]
    co=list(co)
    coe=pd.Series(co, index=sft)
    print coe

    su=co[0] * X_test.ix[te][sft[0]]
    for i in range(1, len(co)):
        su+=co[i]*X_test.ix[te][sft[i]]
    auc = roc_auc_score(y_test.ix[te], su)
    print auc
    coes[auc]=co
'''
print "\nMAX:"
maxc=coes[max(coes.keys())]
su=co[0] * test[sft[0]]
for i in range(1, len(co)):
    su+=co[i]*test[sft[i]]
auc = roc_auc_score(test.LABEL, su)
print auc

print "\nMEAN:"
meanc = pd.DataFrame(coes)
print meanc.ix[0].mean()
su=meanc.ix[0].mean() * test[sft[0]]
faor i in range(1, len(co)):
    tem = meanc.ix[i].mean()
    print tem
    su+=tem*test[sft[i]]
auc = roc_auc_score(test.LABEL, su)
print auc
'''
'''
X_test=test[sft]
y_test=test.LABEL

clf = LogisticRegression(class_weight={1:5, 0:1})
clf.fit(X_test, y_test)

co=clf.coef_
co=co[0]
co=list(co)
coe=pd.Series(co, index=sft)
print coe

su=co[0] * test[sft[0]]
#su=test[sft[0]]
for i in range(1, len(co)):
    su+=co[i] * test[sft[i]]

auc = roc_auc_score(test.LABEL, su)
print auc
#print eval_clf_result(test['LABEL'], su)


mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
fpr, tpr, thresholds = roc_curve(test.LABEL, su)
mean_tpr += interp(mean_fpr, fpr, tpr)
mean_tpr[0] = 0.0
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
'''
