#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:34:41 2019

@author: Broth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale
from imblearn.over_sampling import SMOTE


get_ipython().run_line_magic('matplotlib', 'inline')

dat = pd.read_csv('../../Data/final_vacant_property_inspections.csv')
dat = dat.drop(['Unnamed: 0','APPRAISED_VALUE_CURRENT_TOTAL'], axis = 1)

#create target matrices
X = dat.drop(['CAP_ALIAS','garden_min_dist','bus_stop_min_dist',
               'garden_min_dist','SSL','VACANT_USE','appraised_value_total_diff','WARD','BLDG_TYPE'], axis= 1)


y = dat['CAP_ALIAS']

#create list containing columns that need rescaling
col_to_scale = ['diff_in_sale','PRICE',
                'grocery_min_dist','pubschool_min_dist','chartschool_min_dist',
                'metro_station_min_dist','LATITUDE','LONGITUDE']

#remove nas so that columns can be scaled
X = X.dropna()

#rescale variables
for col in col_to_scale:
    X[col] =  scale(X[col])


#test hyper parameters to see which 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)



#using SMOTE to oversample the minority class by synthetically generating additional samples
#this deals with the imbalance problem in our data
smt = SMOTE(random_state = 123)
X_train_res, y_train_res = smt.fit_sample(X_train, y_train.ravel())

rfmodel = RandomForestClassifier(n_estimators=1000, random_state=1234)
rfmodel.fit(X_train_res, y_train_res)

param_grid = {'max_depth':[3,15],
              'max_features':['auto','log2']
              }

rf_gridsearch = GridSearchCV(estimator=rfmodel, param_grid=param_grid, cv= 10)
rf_gridsearch.fit(X_train, y_train)
best_params = rf_gridsearch.best_params_
print(best_params)

#fit model with best parameters
rfmodel = RandomForestClassifier(n_estimators=1000, max_depth = best_params.get('max_depth'),
                                 max_features = best_params.get('max_features'), random_state = 1234)

rfmodel.fit(X_train_res, y_train_res)
cv10 = cross_val_score(rfmodel, X_train, y_train, cv=10)
cv_mean = np.mean(cv10)
print(cv_mean)


#model.fit(Xtrain, ytrain)
print('Training score: ', rfmodel.score(X_test, y_test))
y_pred = rfmodel.predict(X_test)


### Calculating the AUC for the ROC curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred, pos_label=1)
rates = pd.DataFrame(dict(fpr=false_positive_rate, tpr=true_positive_rate))
roc_auc = auc(rates['fpr'], rates['tpr'])
print('AUC: ', roc_auc)

# plot variable importance
var_imp = pd.DataFrame({'Variable': X.columns, 
                        'Importance': rfmodel.feature_importances_})

var_imp = var_imp.sort_values(by='Importance')
fig2 = plt.figure()
plt.barh(var_imp['Variable'], var_imp['Importance'])
plt.xlabel('Proportion')
plt.ylabel('Variable')
plt.title('Variable Importance Decision Tree Vacant Property')
plt.show()
fig2.savefig('Plots/Variable_Importance_IC.png',dpi=300, bbox_inches = "tight")

#graph of the ROC curve
fig = plt.figure()
plt.plot(rates.fpr, rates.tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve Random Forest Vacant Property')
plt.legend(loc = 'lower right')
plt.show()
fig.savefig('Plots/Random_Forest_ROC_Curve_Vacant_Construction.png',dpi=300, bbox_inches = "tight")

#getting a list of properties with a probability of being vacant>0.5
y_pred = rfmodel.predict(X)
y_pred = pd.DataFrame(y_pred)
y_pred['vacancy_insp'] = y_pred[0]
y_pred = y_pred.drop(0, axis =1)
y_pred.head()

#getting a dataframe of properties with a probability of having illegal construction>0.5
pred_vac=y_pred.join(dat, how='outer')
pred_vac=pred_vac.dropna()
pred_vac=pred_vac[['vacancy_insp', 'SSL', 'LATITUDE','LONGITUDE', 'WARD']]
pred_vac=pred_vac.sort_values(by=['vacancy_insp'], ascending=False)
pred_vac.head()

#save as csv to use for mapping
pred_vac.to_csv('pred_ic_rf.csv')

#pull out all false positives

df = pd.DataFrame()
df["actual"] = y
df["predicted"] = y_pred

#df with only false positives
incorrect = df[df["actual"] != df["predicted"]]
incorrect = incorrect[incorrect['actual'] == 0]
incorrect.head()


