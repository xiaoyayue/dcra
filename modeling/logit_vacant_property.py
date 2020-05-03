#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: Lily Yue and Andrea Quevedo

Logistic Regression Model - Vacant Property

'''


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


#import data
data=pd.read_csv("../../Data/final_vacant_property_inspections.csv", encoding='iso-8859-1',index_col=0)
data.head()

#create a target array and feature matrix
X = data.drop(['CAP_ALIAS', 'SSL','appraised_value_total_diff', 'VACANT_USE'], axis= 1)
y = data['CAP_ALIAS']

#encode categorical variables
lb = LabelEncoder()

col_names= ['BLDG_TYPE','WARD']
for var in col_names:
    X[var]= lb.fit_transform(X[var])

X.head()

#scale variables
scaler = MinMaxScaler(feature_range = (0,1))
X[['APPRAISED_VALUE_CURRENT_TOTAL','diff_in_sale', 'LATITUDE', 'LONGITUDE', 'PRICE', 'garden_min_dist', 'grocery_min_dist','pubschool_min_dist', 'chartschool_min_dist', 'metro_station_min_dist', 'bus_stop_min_dist']] = scaler.fit_transform(X[['APPRAISED_VALUE_CURRENT_TOTAL','diff_in_sale', 'LATITUDE', 'LONGITUDE', 'PRICE', 'garden_min_dist', 'grocery_min_dist','pubschool_min_dist', 'chartschool_min_dist', 'metro_station_min_dist', 'bus_stop_min_dist']])


#check if rescaling is successful
X.head()

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
smt = SMOTE(random_state = 2)
X_train_res, y_train_res = smt.fit_sample(X_train, y_train.ravel())


#define logistic model and check fit
logmodel = LogisticRegression(solver='lbfgs')
logmodel.fit(X_train_res,y_train_res)
print('Training score: ', logmodel.score(X_test, y_test))

#calculate predicted Y using test data
y_pred=logmodel.predict(X_test)

#calculate accuracy rate
accuracy_score(y_test, y_pred)

#ROC curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred, pos_label=1)
rates = pd.DataFrame(dict(fpr=false_positive_rate, tpr=true_positive_rate))
roc_auc = auc(rates['fpr'], rates['tpr'])
print('AUC: ', roc_auc)

#plot ROC curve
plt.plot(rates.fpr, rates.tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve Logit Model: Vacancies')
plt.legend(loc = 'lower right')
plt.savefig("Plots/logit_vacancy.pdf")
plt.show()

#logit model regression output
logit_model = sm.Logit(y, X)
result=logit_model.fit()
print(result.summary2())

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

f = open('logit_vac.tex', 'w')
f.write(beginningtex)
f.write(result.summary2().as_latex())
f.write(endtex)
f.close()

#getting a list of properties with a probability of being vacant>0.5
preds = logmodel.predict_proba(X)
preds = pd.DataFrame(preds)
preds=preds.drop(0, axis=1)
preds.rename(columns={1: 'vacancy_prob'}, inplace=True)
preds=preds.loc[preds['vacancy_prob']>0.5]
preds.head()

#getting a dataframe of properties with a probability of having illegal construction>0.5
pred_vac=preds.join(data, how='outer')
pred_vac=pred_vac.dropna()
pred_vac=pred_vac[['vacancy_prob', 'SSL', 'LATITUDE','LONGITUDE', 'WARD']]
pred_vac=pred_vac.sort_values(by=['vacancy_prob'], ascending=False)
pred_vac.head()

#pull out all false positives
y_pred = logmodel.predict(X)

df = pd.DataFrame()
df["actual"] = y
df["predicted"] = y_pred

#df with only false positives
incorrect = df[df["actual"] != df["predicted"]]
incorrect = incorrect[incorrect['actual'] == 0]
incorrect.head()

#getting a dataframe with probabilities for false positives
fp_lvac=pred_vac.join(incorrect, how='outer')
fp_lvac=fp_lvac.dropna()
fp_lvac=fp_lvac.sort_values(by=['vacancy_prob'], ascending=False)
fp_lvac.head()
fp_lvac

#save as csv to use for mapping
fp_lvac.to_csv('../../Data/fp_lvac.csv')
