#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: Lily Yue and Andrea Quevedo

Logistic Regression Model - Illegal Construction

'''


from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import statsmodels.api as stm
from sklearn.preprocessing import MinMaxScaler

#import data
data=pd.read_csv("../../Data/final_illegal_construction_inspections.csv", encoding='iso-8859-1',index_col=0)
data.head()

#create a target array and feature matrix
X = data.drop(['CAP_ALIAS', 'SSL', 'VACANT_USE', 'appraised_value_total_diff'], axis= 1)
y = data['CAP_ALIAS']

#encode categorical variables
lb = LabelEncoder()

col_names= ['BLDG_TYPE', 'WARD']

for var in col_names:
    X[var]= lb.fit_transform(X[var])

X.head()


#scale variables
scaler = MinMaxScaler(feature_range = (0,1))

X[['APPRAISED_VALUE_CURRENT_TOTAL','diff_in_sale', 'LATITUDE', 'LONGITUDE', 'PRICE', 'garden_min_dist', 'grocery_min_dist','pubschool_min_dist', 'chartschool_min_dist', 'metro_station_min_dist', 'bus_stop_min_dist']] = scaler.fit_transform(X[['APPRAISED_VALUE_CURRENT_TOTAL','diff_in_sale', 'LATITUDE', 'LONGITUDE', 'PRICE', 'garden_min_dist', 'grocery_min_dist','pubschool_min_dist', 'chartschool_min_dist', 'metro_station_min_dist', 'bus_stop_min_dist']])

#check if rescaling is successful
X.head()

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train_res,y_train_res)

#calculate predicted Y using test data
y_pred=logmodel.predict(X_test)

#calculate accuracy rate
accuracy_score(y_test, y_pred)

#ROC curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred, pos_label=1)
rates = pd.DataFrame(dict(fpr=false_positive_rate, tpr=true_positive_rate))
roc_auc = auc(rates['fpr'], rates['tpr'])
print('AUC: ', roc_auc)

#plot roc curve
plt.plot(rates.fpr, rates.tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve Logit Model: Illegal Construction')
plt.legend(loc = 'lower right')
plt.savefig("Plots/logit_constr.pdf")
plt.show()

#logit model regression output
logit_model = stm.Logit(y, X)
result=logit_model.fit()
print(result.summary2())

#getting a list of properties with a probability of having illegal construction >0.5
preds = logmodel.predict_proba(X)
preds = pd.DataFrame(preds)
preds=preds.drop(0, axis=1)
preds.rename(columns={1: 'illegal_const_prob'}, inplace=True)
preds=preds.loc[preds['illegal_const_prob']>0.5]
preds.head()


#getting a dataframe of properties with a probability of having illegal construction>0.5
pred_const=preds.join(data, how='outer')
pred_const=pred_const.dropna()
pred_const=pred_const[['illegal_const_prob', 'SSL', 'LATITUDE','LONGITUDE', 'WARD']]
pred_const=pred_const.sort_values(by=['illegal_const_prob'], ascending=False)
pred_const.head()
pred_const

#getting a lost of false positives
y_pred = logmodel.predict(X)

df = pd.DataFrame()
df["actual"] = y
df["predicted"] = y_pred

#df with only false positives
incorrect = df[df["actual"] != df["predicted"]]
incorrect = incorrect[incorrect['actual'] == 0]
incorrect.head()
incorrect

#getting a dataframe with probabilities for false positives
fp_lillc=pred_const.join(incorrect, how='outer')
fp_lillc=fp_lillc.dropna()
fp_lillc.head()
fp_lillc=fp_lillc.sort_values(by=['illegal_const_prob'], ascending=False)
fp_lillc

#save as csv to use for mapping
fp_lillc.to_csv('../../Data/fp_lillc.csv')
