#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from tester import dump_classifier_and_data
from feature_format import featureFormat
from feature_format import targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('always')




### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',  'salary', 'deferral_payments', 'total_payments', 'loan_advances',
'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',  'expenses',
'exercised_stock_options', 'other', 'long_term_incentive',  'restricted_stock', 'director_fees',
'fraction_exercised_stock_options_poi',  'fraction_exercised_stock_options_npoi',
'from_poi_to_this_person',  'from_this_person_to_poi', 'to_messages',
'from_messages', 'shared_receipt_with_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


#using pandas for data analysis
df_data = pd.DataFrame.from_dict(data_dict, orient='index')
df_data = df_data.replace('NaN', np.nan)

df_data = df_data.drop('email_address', axis=1)


#knowing the database
df_data.info()
print(df_data.head())

# missing values
print("\nMissing Values:")
print(df_data.isnull().sum())


#viewing missing values

print("Viewing missing values")

sns.heatmap(df_data.isnull(), cbar=False)
plt.show(block=True)

#treating missing values
df_data['salary'] = df_data['salary'].fillna(0)
df_data['bonus'] = df_data['bonus'].fillna(0)
df_data['exercised_stock_options'] = df_data['exercised_stock_options'].fillna(0)
df_data['deferral_payments'] = df_data['deferral_payments'].fillna(0)
df_data['total_payments'] = df_data['total_payments'].fillna(0)
df_data['loan_advances'] = df_data['loan_advances'].fillna(0)
df_data['restricted_stock_deferred'] = df_data['restricted_stock_deferred'].fillna(0)
df_data['deferred_income'] = df_data['deferred_income'].fillna(0)
df_data['total_stock_value'] = df_data['total_stock_value'].fillna(0)
df_data['expenses'] = df_data['expenses'].fillna(0)
df_data['other'] = df_data['other'].fillna(0)
df_data['long_term_incentive'] = df_data['long_term_incentive'].fillna(0)
df_data[ 'restricted_stock'] = df_data[ 'restricted_stock'].fillna(0)
df_data['director_fees'] = df_data['director_fees'].fillna(0)

df_data['from_poi_to_this_person'] = df_data['from_poi_to_this_person'].fillna(df_data['from_poi_to_this_person'].median())
df_data['from_this_person_to_poi'] = df_data['from_this_person_to_poi'].fillna(df_data['from_this_person_to_poi'].median())
df_data['to_messages'] = df_data['to_messages'].fillna(df_data['to_messages'].median())
df_data['from_messages'] = df_data['from_messages'].fillna(df_data['from_messages'].median())
df_data['shared_receipt_with_poi'] = df_data['shared_receipt_with_poi'].fillna(df_data['shared_receipt_with_poi'].median())





print("Viewing missing values after treatment")

sns.heatmap(df_data.isnull(), cbar=False)
plt.show(block=True)

### Task 2: Remove outliers

#viewing outliers

print("Viewing outliers")

df_data.plot.scatter(x = 'salary', y = 'bonus')
plt.show(block=True)


#removing outliers

df_data = df_data.drop('TOTAL')
data_dict.pop('TOTAL')



print("Viewing outliers after treatment")


df_data.plot.scatter(x = 'salary', y = 'bonus')
plt.show(block=True)


### Task 3: Create new feature(s)

df_data['poi'] = df_data['poi'].map({True:1, False:0})


sum_poi = sum(df_data['poi'] == 1)
sum_npoi = sum(df_data['poi'] == 0)

print("Number of POIs:", sum_poi)
print("Number of not POIs:", sum_npoi)


eso_poi = df_data.loc[df_data['poi'] == 1]
eso_npoi = df_data.loc[df_data['poi'] == 0]

sum_eso_poi = sum(eso_poi['exercised_stock_options'])
sum_eso_npoi = sum(eso_npoi['exercised_stock_options'])

#as a new characteristic a value was used trying to relate the Average of the values with actions by POI / non-POI
df_data['fraction_exercised_stock_options_poi'] = df_data['exercised_stock_options']/float(sum_eso_poi/sum_poi)
df_data['fraction_exercised_stock_options_npoi'] = df_data['exercised_stock_options']/float(sum_eso_npoi/sum_npoi)

### Store to my_dataset for easy export below.

my_dataset = df_data.to_dict('index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


#seleciton features

from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)
features = selector.fit_transform(features, labels)
feature_indices = selector.get_support(indices=True)


features_list2 = ['poi']
for index in feature_indices:
    features_list2.append(features_list[index+1])

features_list = features_list2

print("Features list:", features_list)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.metrics import fbeta_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test  = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print "Training set has {} samples.".format(len(features_train))
print "Testing set has {} samples.".format(len(features_test))




### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


#classifiers

#clf_Ada = AdaBoostClassifier(random_state=42)
#clf_Ada.fit(features_train, labels_train)
#pred = clf_Ada.predict(features_test)

#clf_Ada2 = AdaBoostClassifier(algorithm = 'SAMME', n_estimators = 5)
#clf_Ada2.fit(features_train, labels_train)
#pred = clf_Ada2.predict(features_test)

#clf_lr =  LogisticRegression(random_state=42)
#clf_lr.fit(features_train, labels_train)
#pred = clf_lr.predict(features_test)


#clf_Ada3 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=42), random_state=42)
#parameters = {'n_estimators': [2,4,6],
            #  'learning_rate':[0.1, 0.5, 1., 10],
            #  'base_estimator__min_samples_split' : np.arange(2, 8, 2),
            #  'base_estimator__max_depth' : np.arange(1, 4, 1)
            # }
#scorer = make_scorer(fbeta_score, beta=0.5)
#grid_obj = GridSearchCV(clf_Ada3, parameters, scorer)
#grid_fit = grid_obj.fit(features_train,labels_train)
#best_clf = grid_fit.best_estimator_
#pred = best_clf.predict(features_test)

clf_gnb =  GaussianNB()
parameters = {'priors': [None]}
scorer = make_scorer(fbeta_score, beta=0.5)
grid_obj = GridSearchCV(clf_gnb, parameters, scorer)
grid_fit = grid_obj.fit(features_train,labels_train)
best_clf = grid_fit.best_estimator_
pred = best_clf.predict(features_test)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "F1 Score: ", f1_score(labels_test, pred)
print "Accuracy: ", accuracy_score(labels_test, pred)
print "Precision Score: ", precision_score(labels_test, pred)
print "Recall Score: ", recall_score(labels_test, pred)



#Select a classifier as final

clf = best_clf


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
