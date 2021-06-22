import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy
from sklearn import metrics

#### Seed
import random
seed = 42
np.random.seed(seed)







############ Read CSV file
dataFr = pd.read_csv("Dataset_BP_45feature.csv")
dataFr.head()
dataFr.shape
df = dataFr
print(df.head())
print(df.shape)
print(df.isnull().any().any())
print(df.isnull().sum().sum())
print(df.isnull().any().any())
print(df.isnull().sum())

from sklearn.preprocessing import MinMaxScaler

y = df.iloc[:, 49]
X = df.iloc[:, 1:48]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


fs = SelectKBest(score_func=f_regression, k=16)
# apply feature selection
X_selected = fs.fit_transform(X, y)
print(X_selected)


import seaborn as sns
data = df.iloc[:,1:49]
# data = df.drop(df.columns[[0, 48]], axis=1) 
corr = data.corr()
sns.heatmap(corr)


columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] > 0.5:
            if columns[j]:
                columns[j] = False
     
selected_columns = data.columns[columns]



# data = data[selected_columns]
# selected_columns = selected_columns[1:].values

# import csv
# with open('CFSDP.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(selected_columns)
#     for i in range(0, data.shape[0]):
#          writer.writerow(data.iloc[i,:])
        

from sklearn.feature_selection import RFE
nof_list=np.arange(1,47)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,n_features_to_select=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))



cols = list(X.columns)
model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, n_features_to_select=31)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)



from skfeature.function.similarity_based import reliefF
y = df.iloc[:, 49].values
X = df.iloc[:, 1:48].values

# n_samples, n_features = X.shape

# score = reliefF.reliefF(X, y)
# idx = reliefF.feature_ranking(score)
# col = list(df.columns)
# selected_features = X[:, idx[0:18]]

# from sklearn.ensemble import RandomForestRegressor 
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import RFECV
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score

# from sklearn import metrics

# rfp = {'max_depth': 7,
#        'min_samples_leaf': 2,
#        'min_samples_split': 5,
#        'n_estimators': 10,
#        'random_state': 42}

# def RFR():
#     rnr=RandomForestRegressor(**rfp)
#     return rnr




# import seaborn as sns
# data = df.iloc[:,1:49]
# # data = df.drop(df.columns[[0, 48]], axis=1) 
# corr = data.corr()
# sns.heatmap(corr)

# columns = np.full((corr.shape[0],), True, dtype=bool)
# for i in range(corr.shape[0]):
#     for j in range(i+1, corr.shape[0]):
#         if corr.iloc[i,j] > 0.5:
#             if columns[j]:
#                 columns[j] = False
# selected_columns = data.columns[columns]

# data = data[selected_columns]

# y = data.iloc[:, 17]
# X = data.iloc[:, 0:17]

# selected_columns = selected_columns[1:].values

# cv = KFold(n_splits=5, random_state=42, shuffle=True)

# rfecv = RFECV(estimator=RFR(), step=1, cv=cv, scoring='neg_mean_absolute_error')
# rfecv.fit(X, y)

# print('Optimal number of features: {}'.format(rfecv.n_features_))


# plt.figure(figsize=(16, 9))
# plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
# plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
# plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

# plt.show()

# X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
# dset = pd.DataFrame()
# dset['attr'] = X.columns
# dset['importance'] = rfecv.estimator_.feature_importances_
# dset = dset.sort_values(by='importance', ascending=False)
# plt.figure(figsize=(16, 14))
# plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
# plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
# plt.xlabel('Importance', fontsize=14, labelpad=20)
# plt.show()



# selected_columns =list(dset['attr'])

# RFE = X[selected_columns]
# RFE['Diastolic Blood Pressure(mmHg)'] = df['Diastolic Blood Pressure(mmHg)']

# selected_columns.append('Diastolic Blood Pressure(mmHg)')
# import csv
# with open('RFEDP.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(selected_columns)
#     for i in range(0, RFE.shape[0]):
#           writer.writerow(RFE.iloc[i,:])








# import seaborn as sns
# data = df.iloc[:,1:49]
# # data = df.drop(df.columns[[0, 48]], axis=1) 
# corr = data.corr()
# sns.heatmap(corr)

# columns = np.full((corr.shape[0],), True, dtype=bool)
# for i in range(corr.shape[0]):
#     for j in range(i+1, corr.shape[0]):
#         if corr.iloc[i,j] > 0.5:
#             if columns[j]:
#                 columns[j] = False
# selected_columns = data.columns[columns]

# data = data[selected_columns]

# y = data.iloc[:, 16]
# X = data.iloc[:, 0:16]

# selected_columns = selected_columns[1:].values

# cv = KFold(n_splits=5, random_state=42, shuffle=True)

# rfecv = RFECV(estimator=RFR(), step=1, cv=cv, scoring='neg_mean_absolute_error')
# rfecv.fit(X, y)

# print('Optimal number of features: {}'.format(rfecv.n_features_))


# plt.figure(figsize=(16, 9))
# plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
# plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
# plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

# plt.show()

# X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
# dset = pd.DataFrame()
# dset['attr'] = X.columns
# dset['importance'] = rfecv.estimator_.feature_importances_
# dset = dset.sort_values(by='importance', ascending=False)
# plt.figure(figsize=(16, 14))
# plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
# plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
# plt.xlabel('Importance', fontsize=14, labelpad=20)
# plt.show()



# selected_columns =list(dset['attr'])

# RFE = X[selected_columns]
# RFE['Systolic Blood Pressure(mmHg)'] = df['Systolic Blood Pressure(mmHg)']

# selected_columns.append('Systolic Blood Pressure(mmHg)')
# import csv
# with open('RFESP.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(selected_columns)
#     for i in range(0, RFE.shape[0]):
#           writer.writerow(RFE.iloc[i,:])






