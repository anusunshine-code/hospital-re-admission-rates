#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:53:13 2018

@author: ANU PAHLAJANI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


d = pd.read_csv('C:/Users/Anu/Desktop/dataset_diabetes/diabetic_data.csv')

#drop attributes with a large percentage of unknown
d.drop('weight', axis=1, inplace=True)
d.drop('payer_code', axis=1, inplace=True)
d.drop('medical_specialty', axis=1, inplace=True)

#drop rows with missing information
d.replace(['?'], np.nan, inplace=True)
d.dropna(inplace=True)

#drop attibutes not relevent to the readmission rate
d.drop('encounter_id', axis=1, inplace=True)
d.drop('patient_nbr', axis=1, inplace=True)

#drop attibutes that only contain one value ('No')
d.drop(['citoglipton','examide'], axis=1, inplace=True)

#change race from categorical to numeric
d['race'] = d['race'].replace('AfricanAmerican', 1)
d['race'] = d['race'].replace('Asian', 2)
d['race'] = d['race'].replace('Caucasian', 3)
d['race'] = d['race'].replace('Hispanic', 4)
d['race'] = d['race'].replace('Other', 5)

#change gender from categorical to binary
d.replace(['Unknown/Invalid'], np.nan, inplace=True)
d.dropna(inplace=True)
d['gender'] = d['gender'].replace('Male', 1)
d['gender'] = d['gender'].replace('Female', 0)

#change age from categorical to numeric
age_dict = {'[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35, '[40-50)':45,
'[50-60)':55, '[60-70)':65, '[70-80)':75, '[80-90)':85, '[90-100)':95}
d['age'] = d.age.map(age_dict)
d['age'] = d['age'].astype('int64')

#change medicines from categorical to numeric
meds = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
        'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 
        'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 
        'glyburide-metformin', 'tolazamide', 'metformin-pioglitazone',
        'metformin-rosiglitazone', 'glimepiride-pioglitazone', 
        'glipizide-metformin', 'troglitazone', 'tolbutamide', 'acetohexamide']

for col in meds:
    d[col] = d[col].replace('No', 0)
    d[col] = d[col].replace('Steady', 1)
    d[col] = d[col].replace('Up', 1)
    d[col] = d[col].replace('Down', 1)

#creat a new attribute to record the total number of medicines used
d.insert(40,'numofmeds',0)
for col in meds:
    d['numofmeds'] = d['numofmeds'] + d[col]

# create a duplicate of the diagnosis column
d.insert(14,'level1_diag1',0)
d['level1_diag1'] = d['diag_1']

# disease codes starting with V or E are in “other” category; so recode them to 0
d.loc[d['diag_1'].str.contains('V'), ['level1_diag1']] = 0
d.loc[d['diag_1'].str.contains('E'), ['level1_diag1']] = 0
d['level1_diag1'] = d['level1_diag1'].astype(float)

# iterate and recode disease codes between certain ranges to certain categories
for index, row in d.iterrows():
    if (row['level1_diag1'] >= 390 and row['level1_diag1'] < 460) or (np.floor(row['level1_diag1']) == 785):
        d.loc[index, 'level1_diag1'] = 1
    elif (row['level1_diag1'] >= 460 and row['level1_diag1'] < 520) or (np.floor(row['level1_diag1']) == 786):
        d.loc[index, 'level1_diag1'] = 2
    elif (row['level1_diag1'] >= 520 and row['level1_diag1'] < 580) or (np.floor(row['level1_diag1']) == 787):
        d.loc[index, 'level1_diag1'] = 3
    elif (np.floor(row['level1_diag1']) == 250):
        d.loc[index, 'level1_diag1'] = 4
    elif (row['level1_diag1'] >= 800 and row['level1_diag1'] < 1000):
        d.loc[index, 'level1_diag1'] = 5
    elif (row['level1_diag1'] >= 710 and row['level1_diag1'] < 740):
        d.loc[index, 'level1_diag1'] = 6
    elif (row['level1_diag1'] >= 580 and row['level1_diag1'] < 630) or (np.floor(row['level1_diag1']) == 788):
        d.loc[index, 'level1_diag1'] = 7
    elif (row['level1_diag1'] >= 140 and row['level1_diag1'] < 240):
        d.loc[index, 'level1_diag1'] = 8
    else:
        d.loc[index, 'level1_diag1'] = 0
# convert this variable to float type to enable computations later
d['level1_diag1'] = d['level1_diag1'].astype(float)

#change change and diabetesmed to numeric
d['change'] = d['change'].replace('Ch', 1)
d['change'] = d['change'].replace('No', 0)
d['diabetesMed'] = d['diabetesMed'].replace('Yes', 1)
d['diabetesMed'] = d['diabetesMed'].replace('No', 0)

#drop diag_1, diag_2, diag_3
d.drop(['diag_1','diag_2','diag_3'], axis=1, inplace=True)

#change A1Cresult and max_glu_serum to numeric
d['A1Cresult'] = d['A1Cresult'].replace('>8', 3)
d['A1Cresult'] = d['A1Cresult'].replace('>7', 2)
d['A1Cresult'] = d['A1Cresult'].replace('Norm', 1)
d['A1Cresult'] = d['A1Cresult'].replace('None', 0)
d['max_glu_serum'] = d['max_glu_serum'].replace('>300', 3)
d['max_glu_serum'] = d['max_glu_serum'].replace('>200', 2)
d['max_glu_serum'] = d['max_glu_serum'].replace('Norm', 1)
d['max_glu_serum'] = d['max_glu_serum'].replace('None', 0)

#change readmission rate to numeric
d['readmitted'] = d['readmitted'].replace('>30', 0)
d['readmitted'] = d['readmitted'].replace('<30', 1)
d['readmitted'] = d['readmitted'].replace('NO', 0)

#shuffle data
d = d.sample(n = len(d), random_state = 0)
d = d.reset_index(drop = True)

#slipt attibutes and results
X_pre = d.iloc[:,:-1].values
y_pre = d.iloc[:,-1].values

#feature selection (15 attributes)
selector = SelectKBest(chi2, k= 15).fit(X_pre,y_pre)
#X_train = selector.transform(X_train) # not needed to get the score
scores = selector.scores_

plt.bar(range(len(scores)), scores)
plt.xticks(range(len(d.columns.values)), d.columns.values, rotation='vertical')
plt.show()

high_15 = np.take(d.columns.values[:-1], (-scores).argsort()[:15])
feature_d = d[list(high_15)]

#split train(80%) and test(20%) data
feature_d_test = feature_d.sample(frac = 0.20, random_state = 0)
feature_d_train = feature_d.drop(feature_d_test.index)

X_train = feature_d_train.iloc[:,:-1].values
y_train = feature_d_train.iloc[:,-1].values
X_test = feature_d_test.iloc[:,:-1].values
y_test = feature_d_test.iloc[:,-1].values


#Logistic regression
print('Logistic regression')
logreg = LogisticRegression(fit_intercept=True, penalty='l2', solver = 'lbfgs', max_iter = 4000)
print("Cross Validation Score: {:.2%}".format(np.mean(cross_val_score(logreg, X_train, y_train, cv=10))))
logreg.fit(X_train, y_train)
print("Dev Set score: {:.2%}".format(logreg.score(X_test, y_test)))
Y_test_predict = logreg.predict(X_test)
pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(Y_test_predict, name = 'Predict'), margins = True)
print('The confusion matrix is')
print(confusion_matrix(y_test,Y_test_predict))
#tn0, fp0, fn0, tp0 = confusion_matrix(y_test,Y_test_predict)
print("Accuracy is {0:.2f}".format(accuracy_score(y_test, Y_test_predict)))
print("Precision is {0:.2f}".format(precision_score(y_test, Y_test_predict )))
print("Recall is {0:.2f}".format(recall_score(y_test, Y_test_predict )))
print(' ')

#Decision tree
print('Decision tree')
dte = DecisionTreeClassifier(max_depth = 20, random_state = 0,criterion = 'gini')
print("Cross Validation score: {:.2%}".format(np.mean(cross_val_score(dte, X_train, y_train, cv=10))))
dte.fit(X_train, y_train)
print("Dev Set score: {:.2%}".format(dte.score(X_test, y_test)))
Y_test_predict1 = dte.predict(X_test)
pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(Y_test_predict1, name = 'Predict'), margins = True)
print('The confusion matrix is')
print(confusion_matrix(y_test,Y_test_predict1))
print("Accuracy is {0:.2f}".format(accuracy_score(y_test, Y_test_predict1)))
print("Precision is {0:.2f}".format(precision_score(y_test, Y_test_predict1 )))
print("Recall is {0:.2f}".format(recall_score(y_test, Y_test_predict1 )))
print(' ')

#Naive Bayes
print('Naive Bayes')
nb = GaussianNB()
print("Cross Validation score: {:.2%}".format(np.mean(cross_val_score(nb, X_train, y_train, cv=10))))
nb.fit(X_train, y_train)
print("Dev Set score: {:.2%}".format(nb.score(X_test, y_test)))
Y_test_predict2 = nb.predict(X_test)
pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(Y_test_predict2, name = 'Predict'), margins = True)
print('The confusion matrix is')
print(confusion_matrix(y_test,Y_test_predict2))
print("Accuracy is {0:.2f}".format(accuracy_score(y_test, Y_test_predict2)))
print("Precision is {0:.2f}".format(precision_score(y_test, Y_test_predict2 )))
print("Recall is {0:.2f}".format(recall_score(y_test, Y_test_predict2 )))
print(' ')

#Neural network
print('Neural network')
neuralnet = MLPClassifier()
print("Cross Validation score: {:.2%}".format(np.mean(cross_val_score(neuralnet, X_train, y_train, cv=10))))
neuralnet.fit(X_train, y_train)
print("Dev Set score: {:.2%}".format(neuralnet.score(X_test, y_test)))
Y_test_predict3 = neuralnet.predict(X_test)
pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(Y_test_predict3, name = 'Predict'), margins = True)
print('The confusion matrix is')
print(confusion_matrix(y_test,Y_test_predict3))

print("Accuracy is {0:.2f}".format(accuracy_score(y_test, Y_test_predict3)))
print("Precision is {0:.2f}".format(precision_score(y_test, Y_test_predict3 )))
print("Recall is {0:.2f}".format(recall_score(y_test, Y_test_predict3 )))
print(' ')

#Random forest
print('Random forest')
randomfor = RandomForestClassifier()
print("Cross Validation score: {:.2%}".format(np.mean(cross_val_score(randomfor, X_train, y_train, cv=10))))
randomfor.fit(X_train, y_train)
print("Dev Set score: {:.2%}".format(randomfor.score(X_test, y_test)))
Y_test_predict4 = randomfor.predict(X_test)
pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(Y_test_predict4, name = 'Predict'), margins = True)
print('The confusion matrix is')
print(confusion_matrix(y_test,Y_test_predict4))
print("Accuracy is {0:.2f}".format(accuracy_score(y_test, Y_test_predict4)))
print("Precision is {0:.2f}".format(precision_score(y_test, Y_test_predict4 )))
print("Recall is {0:.2f}".format(recall_score(y_test, Y_test_predict4 )))
print(' ')
