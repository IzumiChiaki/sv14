# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:25:50 2019

@author: Chiaki
"""

import sys
sys.path.append("/SV14")

from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA 
from sklearn.model_selection import LeaveOneOut
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from minepy import MINE
mine = MINE(alpha=0.6, c=15, est='mic_approx')

#1 import data
data = pd.read_csv('../dataFiles/SV14.csv')
df = data.drop(columns=['ID','PU#1','PU#2','PU#3','FC#1','FC#2','FC#3','AC#1','AC#2','AC#3','EP#1','EP#2','EP#3'])

#2 plot each column
dfA = df.drop(columns=['Location'])                                              
i = 1
plt.figure(figsize=(9, 9))
for group in range(dfA.values.shape[1]):  
    plt.subplot(6, 3, i)
    plt.plot(dfA.values[:, group])
    plt.title(dfA.columns[group], y=0.5, loc='right')
    i += 1
    plt.show()
plt.savefig('../results/EachColumnMap.png')

#3 compute pearson and mic
def printMic(x,y):
    mine.compute_score(x,y)
    return mine.mic()
pr = np.array([[pearsonr(df.values[:,i+1], df.values[:,15+j])[0] for j in range(4)] for i in range(df.values.shape[1]-5)])
micr = np.array([[printMic(df.values[:,i+1], df.values[:,15+j]) for j in range(4)] for i in range(df.values.shape[1]-5)])

environmentstr = list(df)[1:15]
coatingstr = list(df)[-4:]
pearson = pd.DataFrame(pr, index=environmentstr, columns=coatingstr)
mic = pd.DataFrame(micr, index=environmentstr, columns=coatingstr)

##3.1 plot pearson heatmap
df_environment = df[environmentstr].corr('pearson')
plt.subplots(figsize=(9, 9))
sns.heatmap(df_environment, annot=True,annot_kws={'size':8}, vmax=1, square=True, cmap="Blues")
plt.savefig('../results/PearsonHeatmap_Environment.png')
plt.show()

dfData_list = []
dfData = []
for i in range(4):
    dfData_list.append(df.drop(columns=coatingstr[:i]+coatingstr[i+1:]))
    dfData.append(dfData_list[i].corr('pearson'))
    plt.subplots(figsize=(9, 9))
    sns.heatmap(dfData[i], annot=True,annot_kws={'size':8}, vmax=1, square=True, cmap="Blues")
    plt.savefig('../results/PearsonHeatmap_'+coatingstr[i]+'.png')
    plt.show()

    
#4 PCA analysis
tofile = '../results/PCAresults.xls' #files path and name

dfPCA = df.drop(columns = ['Location','PU','FC','AC','EP'])
dfPCA_mean = np.mean(dfPCA, axis=0)
std_mat = (dfPCA - dfPCA_mean) / np.std(dfPCA, axis=0, ddof=1)

cov_mat = np.cov(std_mat, rowvar=0)
eig_values, eig_vectors = np.linalg.eig(np.mat(cov_mat))


indices = np.argsort(eig_values)[::-1]   # index of eig_values from small to large
eig_values_sort = eig_values[indices]
eig_vectors_sort = eig_vectors[:, indices]

m = np.dot(std_mat, eig_vectors)
explained_variance_ratio = eig_values / sum(eig_values) # compute contribution rate
explained_variance_ratio_cumulative = np.cumsum(explained_variance_ratio) # compute cumulative contribution rate

writer = pd.ExcelWriter(tofile)

##4.1 write Correlation Matrix
data_df = pd.DataFrame(cov_mat)
data_df.columns = dfPCA.columns[0:]
data_df.index = dfPCA.columns[0:]
data_df.to_excel(writer, 'Correlation Matrix', float_format='%.5f')
writer.save()

eigenvalues = {'component':[], 'value':[], 'difference':[], 'proportion':[], 'cumulative':[]}
for i in range(len(explained_variance_ratio)):
    eigenvalues['component'].append(i + 1)
    eigenvalues['value'].append(eig_values[i])
    if i != len(explained_variance_ratio) - 1:
        eigenvalues['difference'].append(eig_values[i] - eig_values[i + 1])
    else:
        eigenvalues['difference'].append(0)
    eigenvalues['proportion'].append(explained_variance_ratio[i])
    eigenvalues['cumulative'].append(explained_variance_ratio_cumulative[i])

data_df = pd.DataFrame(eigenvalues, columns=eigenvalues.keys())
data_df.to_excel(writer, 'Eigenvalues', index=False, float_format='%.5f')
writer.save()

##4.2 PCA dimensionality reduction
dim = 4
PCA_columns = ['Z' + str(i+1) for i in range(dim)]
pca = PCA(dim)
afterPCA = pca.fit_transform(dfPCA.values)
pr_PCA = np.array([[pearsonr(dfPCA.values[:,i], afterPCA[:,j])[0] for j in range(afterPCA.shape[1])] for i in range(dfPCA.values.shape[1])])
Z_PCA = pd.DataFrame(afterPCA, columns=PCA_columns)
pearson_PCA = pd.DataFrame(pr_PCA, index=environmentstr, columns=PCA_columns)
Z_PCA.to_csv("../results/PrincipalComponents.csv", index=False)
pearson_PCA.to_csv("../results/E-PC_Pearson.csv", index=False)

#5 four-dimension feature and GLR of PU modeling
sample = pd.concat([Z_PCA,df['PU']],axis=1)
loo = LeaveOneOut()
sc = MinMaxScaler(feature_range=(0, 1))
scaled = sc.fit_transform(sample)
PredValue_SVR = []
TrueValue_SVR = []
R2_SVR = []
MAE_train_SVR = [] 
MSE_train_SVR = [] 
RMSE_train_SVR = []
APE_test_SVR = []
MAE_test_SVR = [] 
MSE_test_SVR = [] 
RMSE_test_SVR = []

PredValue_RFR = []
TrueValue_RFR = []
R2_RFR = []
MAE_train_RFR = [] 
MSE_train_RFR = [] 
RMSE_train_RFR = []
APE_test_RFR = []
MAE_test_RFR = [] 
MSE_test_RFR = [] 
RMSE_test_RFR = []

PredValue_LR = []
TrueValue_LR = []
R2_LR = []
MAE_train_LR = [] 
MSE_train_LR = [] 
RMSE_train_LR = []
APE_test_LR = []
MAE_test_LR = [] 
MSE_test_LR = [] 
RMSE_test_LR = []
i = 0
for train_index, test_index in loo.split(scaled):
    trainSet = scaled[train_index]
    testSet = scaled[test_index]
    
    train_X, train_y = trainSet[:,0:4], trainSet[:,-1]
    test_X, test_y = testSet[:,0:4], testSet[:,-1]
    
    clf_SVR = svm.SVR(kernel='rbf',C=1000,gamma=15).fit(train_X,train_y)
    #clf_SVR = svm.SVR(kernel='linear',C=20).fit(train_X,train_y)
    #clf_SVR = svm.SVR(kernel='poly',C=1000, degree=3).fit(train_X,train_y)
    
    clf_RFR = RandomForestRegressor().fit(train_X,train_y)
    #clf_RFR = RandomForestRegressor(n_estimators=100,max_features=2).fit(train_X,train_y)
    
    
    clf_LR= linear_model.LinearRegression().fit(train_X,train_y)
    
    #joblib.dump(clf_SVR, '../results/SVR_train_model_'+str(i+1)+'.m')
    #joblib.dump(clf_RFR, '../results/RFR_train_model_'+str(i+1)+'.m')
    #joblib.dump(clf_LR, '../results/LR_train_model_'+str(i+1)+'.m')
    
    #inverse dataset of SVR
    train_pred_SVR = clf_SVR.predict(train_X)
    test_pred_SVR = clf_SVR.predict(test_X)
    X_SVR = concatenate((train_X, test_X), axis=0) 
    train_pred_SVR = train_pred_SVR.reshape((len(train_pred_SVR), 1))
    test_pred_SVR = test_pred_SVR.reshape((len(test_pred_SVR), 1))
    y_pred_SVR = concatenate((train_pred_SVR, test_pred_SVR), axis=0) #concatenate trainSet and testSet by rows
    inv_y_SVR = concatenate((X_SVR, y_pred_SVR), axis=1) #concatenate sample and label by columns
    inv_y_SVR = sc.inverse_transform(inv_y_SVR) #inverse data
    inv_y_SVR = inv_y_SVR[:,-1] #actual pred_values of GLR in trainSet and testSet
    
    #inverse dataset of RFR
    train_pred_RFR = clf_RFR.predict(train_X)
    test_pred_RFR = clf_RFR.predict(test_X)
    X_RFR = concatenate((train_X, test_X), axis=0) 
    train_pred_RFR = train_pred_RFR.reshape((len(train_pred_RFR), 1))
    test_pred_RFR = test_pred_RFR.reshape((len(test_pred_RFR), 1))
    y_pred_RFR = concatenate((train_pred_RFR, test_pred_RFR), axis=0) #concatenate trainSet and testSet by rows
    inv_y_RFR = concatenate((X_RFR, y_pred_RFR), axis=1) #concatenate sample and label by columns
    inv_y_RFR = sc.inverse_transform(inv_y_RFR) #inverse data
    inv_y_RFR = inv_y_RFR[:,-1] #actual pred_values of GLR in trainSet and testSet
    
    #inverse dataset of LR
    train_pred_LR = clf_LR.predict(train_X)
    test_pred_LR = clf_LR.predict(test_X)
    X_LR = concatenate((train_X, test_X), axis=0) 
    train_pred_LR = train_pred_LR.reshape((len(train_pred_LR), 1))
    test_pred_LR = test_pred_LR.reshape((len(test_pred_LR), 1))
    y_pred_LR = concatenate((train_pred_LR, test_pred_LR), axis=0) #concatenate trainSet and testSet by rows
    inv_y_LR = concatenate((X_LR, y_pred_LR), axis=1) #concatenate sample and label by columns
    inv_y_LR = sc.inverse_transform(inv_y_LR) #inverse data
    inv_y_LR = inv_y_LR[:,-1] #actual pred_values of GLR in trainSet and testSet
    
    train_y = train_y.reshape((len(train_y), 1))
    test_y = test_y.reshape((len(test_y), 1))
    y_true = concatenate((train_y, test_y), axis=0)
    #SVR
    inv_yt_SVR = concatenate((X_SVR, y_true), axis=1)
    inv_yt_SVR = sc.inverse_transform(inv_yt_SVR)
    inv_yt_SVR = inv_yt_SVR[:,-1] #actual true_values of GLR in trainSet and testSet
    #RFR
    inv_yt_RFR = concatenate((X_RFR, y_true), axis=1)
    inv_yt_RFR = sc.inverse_transform(inv_yt_RFR)
    inv_yt_RFR = inv_yt_RFR[:,-1] #actual true_values of GLR in trainSet and testSet
    #LR
    inv_yt_LR = concatenate((X_LR, y_true), axis=1)
    inv_yt_LR = sc.inverse_transform(inv_yt_LR)
    inv_yt_LR = inv_yt_LR[:,-1] #actual true_values of GLR in trainSet and testSet
    
    print('No.'+str(i+1))
    #SVR
    plt.figure()
    values = list(range(0,14))
    plt.plot(values[0:13],inv_yt_SVR[:-1], label='train_True')
    plt.plot(values[0:13],inv_y_SVR[:-1], label='train_prediction')
    #plt.scatter(values[13],inv_yt_SVR[13], label='test_True')
    plt.scatter(values[13],inv_yt_SVR[13])
    #plt.scatter(values[13],inv_y_SVR[13], label='test_prediction')
    plt.scatter(values[13],inv_y_SVR[13])
    plt.title("SVR_Prediction")
    plt.xlabel('region')
    plt.ylabel('actual_Scaled')
    plt.legend()
    plt.show()
    
    print('Pred:', inv_y_SVR[-1])
    print('True:', inv_yt_SVR[-1])
    print('APE_test:',np.abs(inv_yt_SVR[-1]-inv_y_SVR[-1])/inv_yt_SVR[-1])
    print('MAE_test:', mean_absolute_error([inv_yt_SVR[-1]], [inv_y_SVR[-1]]))
    print('MSE_test:', mean_squared_error([inv_yt_SVR[-1]], [inv_y_SVR[-1]]))
    print('RMSE_test:', np.sqrt(mean_squared_error([inv_yt_SVR[-1]], [inv_y_SVR[-1]])))
    print('MAE_train:', mean_absolute_error(train_y, train_pred_SVR))
    print('MSE_train:', mean_squared_error(train_y, train_pred_SVR))
    print('RMSE_train:', np.sqrt(mean_squared_error(train_y, train_pred_SVR)))
    print('R2_train:', r2_score(train_y, train_pred_SVR))

    #RFR
    plt.figure()
    values = list(range(0,14))
    plt.plot(values[0:13],inv_yt_RFR[:-1], label='train_True')
    plt.plot(values[0:13],inv_y_RFR[:-1], label='train_prediction')
    #plt.scatter(values[13],inv_yt_RFR[13], label='test_True')
    plt.scatter(values[13],inv_yt_RFR[13])
    #plt.scatter(values[13],inv_y_RFR[13], label='test_prediction')
    plt.scatter(values[13],inv_y_RFR[13])
    plt.title("RFR_Prediction")
    plt.xlabel('region')
    plt.ylabel('actual_Scaled')
    plt.legend()
    plt.show()
    
    print('Pred:', inv_y_RFR[-1])
    print('True:', inv_yt_RFR[-1])
    print('APE_test:',np.abs(inv_yt_RFR[-1]-inv_y_RFR[-1])/inv_yt_RFR[-1])
    print('MAE_test:', mean_absolute_error([inv_yt_RFR[-1]], [inv_y_RFR[-1]]))
    print('MSE_test:', mean_squared_error([inv_yt_RFR[-1]], [inv_y_RFR[-1]]))
    print('RMSE_test:', np.sqrt(mean_squared_error([inv_yt_RFR[-1]], [inv_y_RFR[-1]])))
    print('MAE_train:', mean_absolute_error(train_y, train_pred_RFR))
    print('MSE_train:', mean_squared_error(train_y, train_pred_RFR))
    print('RMSE_train:', np.sqrt(mean_squared_error(train_y, train_pred_RFR)))
    print('R2_train:', r2_score(train_y, train_pred_RFR))
    
    #LR
    plt.figure()
    values = list(range(0,14))
    plt.plot(values[0:13],inv_yt_LR[:-1], label='train_True')
    plt.plot(values[0:13],inv_y_LR[:-1], label='train_prediction')
    #plt.scatter(values[13],inv_yt_LR[13], label='test_True')
    plt.scatter(values[13],inv_yt_LR[13])
    #plt.scatter(values[13],inv_y_LR[13], label='test_prediction')
    plt.scatter(values[13],inv_y_LR[13])
    plt.title("LR_Prediction")
    plt.xlabel('region')
    plt.ylabel('actual_Scaled')
    plt.legend()
    plt.show()
    
    print('Pred:', inv_y_LR[-1])
    print('True:', inv_yt_LR[-1])
    print('APE_test:',np.abs(inv_yt_LR[-1]-inv_y_LR[-1])/inv_yt_LR[-1])
    print('MAE_test:', mean_absolute_error([inv_yt_LR[-1]], [inv_y_LR[-1]]))
    print('MSE_test:', mean_squared_error([inv_yt_LR[-1]], [inv_y_LR[-1]]))
    print('RMSE_test:', np.sqrt(mean_squared_error([inv_yt_LR[-1]], [inv_y_LR[-1]])))
    print('MAE_train:', mean_absolute_error(train_y, train_pred_LR))
    print('MSE_train:', mean_squared_error(train_y, train_pred_LR))
    print('RMSE_train:', np.sqrt(mean_squared_error(train_y, train_pred_LR)))
    print('R2_train:', r2_score(train_y, train_pred_LR))
    print('============================================================')
    
    #SVR
    PredValue_SVR.append(inv_y_SVR[-1])
    TrueValue_SVR.append(inv_yt_SVR[-1])
    R2_SVR.append(r2_score(train_y, train_pred_SVR))
    MAE_train_SVR.append(mean_absolute_error(train_y, train_pred_SVR))
    MSE_train_SVR.append(mean_squared_error(train_y, train_pred_SVR))
    RMSE_train_SVR.append(np.sqrt(mean_squared_error(train_y, train_pred_SVR)))
    APE_test_SVR.append(np.abs(inv_yt_SVR[-1]-inv_y_SVR[-1])/inv_yt_SVR[-1])
    MAE_test_SVR.append(mean_absolute_error([inv_yt_SVR[-1]], [inv_y_SVR[-1]]))
    MSE_test_SVR.append(mean_squared_error([inv_yt_SVR[-1]], [inv_y_SVR[-1]]))
    RMSE_test_SVR.append(np.sqrt(mean_squared_error([inv_yt_SVR[-1]], [inv_y_SVR[-1]])))
    #RFR
    PredValue_RFR.append(inv_y_RFR[-1])
    TrueValue_RFR.append(inv_yt_RFR[-1])
    R2_RFR.append(r2_score(train_y, train_pred_RFR))
    MAE_train_RFR.append(mean_absolute_error(train_y, train_pred_RFR))
    MSE_train_RFR.append(mean_squared_error(train_y, train_pred_RFR))
    RMSE_train_RFR.append(np.sqrt(mean_squared_error(train_y, train_pred_RFR)))
    APE_test_RFR.append(np.abs(inv_yt_RFR[-1]-inv_y_RFR[-1])/inv_yt_RFR[-1])
    MAE_test_RFR.append(mean_absolute_error([inv_yt_RFR[-1]], [inv_y_RFR[-1]]))
    MSE_test_RFR.append(mean_squared_error([inv_yt_RFR[-1]], [inv_y_RFR[-1]]))
    RMSE_test_RFR.append(np.sqrt(mean_squared_error([inv_yt_RFR[-1]], [inv_y_RFR[-1]])))
    #LR
    PredValue_LR.append(inv_y_LR[-1])
    TrueValue_LR.append(inv_yt_LR[-1])
    R2_LR.append(r2_score(train_y, train_pred_LR))
    MAE_train_LR.append(mean_absolute_error(train_y, train_pred_LR))
    MSE_train_LR.append(mean_squared_error(train_y, train_pred_LR))
    RMSE_train_LR.append(np.sqrt(mean_squared_error(train_y, train_pred_LR)))
    APE_test_LR.append(np.abs(inv_yt_LR[-1]-inv_y_LR[-1])/inv_yt_LR[-1])
    MAE_test_LR.append(mean_absolute_error([inv_yt_LR[-1]], [inv_y_LR[-1]]))
    MSE_test_LR.append(mean_squared_error([inv_yt_LR[-1]], [inv_y_LR[-1]]))
    RMSE_test_LR.append(np.sqrt(mean_squared_error([inv_yt_LR[-1]], [inv_y_LR[-1]])))
    
    i += 1
    
joblib.dump(clf_SVR, '../results/SVR_train_model.m')
joblib.dump(clf_RFR, '../results/RFR_train_model.m')
joblib.dump(clf_LR, '../results/LR_train_model.m')

#5.1 export data
dfSVR = pd.DataFrame({'id':[i+1 for i in range(len(R2_SVR))],\
                            'PredValue':PredValue_SVR,'TrueValue':TrueValue_SVR,\
                            'APE_test':APE_test_SVR,'MAE_test':MAE_test_SVR,\
                            'MSE_test':MSE_test_SVR,'RMSE_test':RMSE_test_SVR,\
                            'MAE_train':MAE_train_SVR,'MSE_train':MSE_train_SVR,\
                            'RMSE_train':RMSE_train_SVR,'R2_train':R2_SVR,})
dfRFR = pd.DataFrame({'id':[i+1 for i in range(len(R2_RFR))],\
                            'PredValue':PredValue_RFR,'TrueValue':TrueValue_RFR,\
                            'APE_test':APE_test_RFR,'MAE_test':MAE_test_RFR,\
                            'MSE_test':MSE_test_RFR,'RMSE_test':RMSE_test_RFR,\
                            'MAE_train':MAE_train_RFR,'MSE_train':MSE_train_RFR,\
                            'RMSE_train':RMSE_train_RFR,'R2_train':R2_RFR,})
dfLR = pd.DataFrame({'id':[i+1 for i in range(len(R2_LR))],\
                            'PredValue':PredValue_LR,'TrueValue':TrueValue_LR,\
                            'APE_test':APE_test_LR,'MAE_test':MAE_test_LR,\
                            'MSE_test':MSE_test_LR,'RMSE_test':RMSE_test_LR,\
                            'MAE_train':MAE_train_LR,'MSE_train':MSE_train_LR,\
                            'RMSE_train':RMSE_train_LR,'R2_train':R2_LR,})
dfSVR.to_csv("../results/SVRestimate.csv", index=False)
dfRFR.to_csv("../results/RFRestimate.csv", index=False)
dfLR.to_csv("../results/LRestimate.csv", index=False)

#5.1 compute mean of Evaluation
MAPE_test_list = [np.mean(APE_test_SVR), np.mean(APE_test_RFR), np.mean(APE_test_LR)]
MAE_test_list = [np.mean(MAE_test_SVR), np.mean(MAE_test_RFR), np.mean(MAE_test_LR)]
MSE_test_list = [np.mean(MSE_test_SVR), np.mean(MSE_test_RFR), np.mean(MSE_test_LR)]
RMSE_test_list = [np.mean(RMSE_test_SVR), np.mean(RMSE_test_RFR), np.mean(RMSE_test_LR)]
MAE_train_list = [np.mean(MAE_train_SVR), np.mean(MAE_train_RFR), np.mean(MAE_train_LR)]
MSE_train_list = [np.mean(MSE_train_SVR), np.mean(MSE_train_RFR), np.mean(MSE_train_LR)]
RMSE_train_list = [np.mean(RMSE_train_SVR), np.mean(RMSE_train_RFR), np.mean(RMSE_train_LR)]
R2_train_list = [np.mean(R2_SVR), np.mean(R2_RFR), np.mean(R2_LR)]
print('MAPE_test:', MAPE_test_list)
print('MAE_test:', MAE_test_list)
print('MSE_test:', MSE_test_list)
print('RMSE_test:', RMSE_test_list)
print('MAE_train:', MAE_train_list)
print('MSE_train:', MSE_train_list)
print('RMSE_train:', RMSE_train_list)
print('R2_train:', R2_train_list)
print('============================================================')

#5.2 use model and plot
#import model
clfSVR = joblib.load('../results/SVR_train_model.m')
clfRFR = joblib.load('../results/RFR_train_model.m')
clfLR = joblib.load('../results/LR_train_model.m')

samples = scaled[:,0:4]
labels = scaled[:,-1]

#SVR
labels_pred_SVR = clfSVR.predict(samples)
inv_labels_SVR_actual = concatenate((samples,labels_pred_SVR.reshape(-1,1)), axis=1)
inv_labels_SVR_actual = sc.inverse_transform(inv_labels_SVR_actual)
inv_labels_SVR_actual = inv_labels_SVR_actual[:,-1]

plt.figure()
plt.plot(sample.values[:,-1], label='True')
plt.plot(inv_labels_SVR_actual, label='pred')
plt.title("SVR_Prediction")
plt.xlabel('region')
plt.ylabel('actual_Scaled')
plt.legend()
plt.savefig('../results/SVR_all.png')
plt.show()

print('MSE:', mean_squared_error(sample.values[:,-1], inv_labels_SVR_actual))
print('R2:', r2_score(sample.values[:,-1], inv_labels_SVR_actual))
print('============================================================')

#RFR
labels_pred_RFR = clfRFR.predict(samples)
inv_labels_RFR_actual = concatenate((samples,labels_pred_RFR.reshape(-1,1)), axis=1)
inv_labels_RFR_actual = sc.inverse_transform(inv_labels_RFR_actual)
inv_labels_RFR_actual = inv_labels_RFR_actual[:,-1]

plt.figure()
plt.plot(sample.values[:,-1], label='True')
plt.plot(inv_labels_RFR_actual, label='pred')
plt.title("RFR_Prediction")
plt.xlabel('region')
plt.ylabel('actual_Scaled')
plt.legend()
plt.savefig('../results/RFR_all.png')
plt.show()

print('MSE:', mean_squared_error(sample.values[:,-1], inv_labels_RFR_actual))
print('R2:', r2_score(sample.values[:,-1], inv_labels_RFR_actual))
print('============================================================')

#LR
labels_pred_LR = clfLR.predict(samples)
inv_labels_LR_actual = concatenate((samples,labels_pred_LR.reshape(-1,1)), axis=1)
inv_labels_LR_actual = sc.inverse_transform(inv_labels_LR_actual)
inv_labels_LR_actual = inv_labels_LR_actual[:,-1]

plt.figure()
plt.plot(sample.values[:,-1], label='True')
plt.plot(inv_labels_LR_actual, label='pred')
plt.title("LR_Prediction")
plt.xlabel('region')
plt.ylabel('actual_Scaled')
plt.legend()
plt.savefig('../results/LR_all.png')
plt.show()

print('MSE:', mean_squared_error(sample.values[:,-1], inv_labels_LR_actual))
print('R2:', r2_score(sample.values[:,-1], inv_labels_LR_actual))
print('============================================================')

#ALL
plt.figure()
plt.plot(sample.values[:,-1], label='True')
plt.plot(inv_labels_SVR_actual, label='pred_SVR')
plt.plot(inv_labels_RFR_actual, label='pred_RFR')
plt.plot(inv_labels_LR_actual, label='pred_LR')
plt.title("Prediction")
plt.xlabel('region')
plt.ylabel('actual_Scaled')
plt.legend()
plt.savefig('../results/all.png')
plt.show()
print('MSE_SVR:', mean_squared_error(sample.values[:,-1], inv_labels_SVR_actual))
print('MSE_RFR:', mean_squared_error(sample.values[:,-1], inv_labels_RFR_actual))
print('MSE_LR:', mean_squared_error(sample.values[:,-1], inv_labels_LR_actual))
print('R2_SVR:', r2_score(sample.values[:,-1], inv_labels_SVR_actual))
print('R2_RFR:', r2_score(sample.values[:,-1], inv_labels_RFR_actual))
print('R2:_LR', r2_score(sample.values[:,-1], inv_labels_LR_actual))
print('============================================================')