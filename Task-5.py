import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
ds=pd.read_csv("/content/Advertising.csv") #reading csv file i.e reading dataset
ds.head() #viewing file i.e reading dataset

ds.shape # for finding out number of columns and rows

ds.columns.values.tolist() #column features

ds.info # general information of the dataset

ds.describe() #for statistical analysis

ds.isnull().sum() #checking out null values

import matplotlib.pyplot as plt 
import seaborn as sns
#importing these libraries for exploratory data analysis

# for checking outliers
fig, axs = plt.subplots(3,figsize=(5,5))
plt1=sns.boxplot(ds['TV'],ax=axs[0])
plt2=sns.boxplot(ds['Newspaper'],ax=axs[1])
plt3=sns.boxplot(ds['Radio'],ax=axs[2])
plt.tight_layout()

sns.distplot(ds['Newspaper'])

iqr = ds.Newspaper.quantile(0.75) - ds.Newspaper.quantile(0.25)

lower_bridge = ds['Newspaper'].quantile(0.25) - (iqr*1.5)
upper_bridge = ds['Newspaper'].quantile(0.25) + (iqr*1.5)
print(lower_bridge)
print(upper_bridge)

copieddata=ds.copy()

copieddata.loc[copieddata['Newspaper']>=93, 'Newspaper'] = 93

sns.boxplot(copieddata['Newspaper'])

sns.boxplot(copieddata['Sales'])

sns.pairplot(copieddata,x_vars=['TV','Newspaper','Radio'],y_vars='Sales',height=5,aspect=1,kind='scatter')
plt.show()

sns.heatmap(copieddata.corr(),cmap="YlGnBu",annot=True)
plt.show()

important_features = list(ds.corr()['Sales'][(ds.corr()['Sales']>+0.5)|(ds.corr()['Sales']<-0.5)].index)#here we are selecting important features having correlation of 0.5 and -1.5

print(important_features)

#declaring dependent and independent variable or features
x = copieddata['TV']
y = copieddata['Sales']

x = x.values.reshape(-1,1) 

x # for array values

y 

print(x.shape,y.shape)

#for splitting data we use train and test method
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

#importing the modal algorithms
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

#using k neighborRegressor
knn = KNeighborsRegressor().fit(x_train,y_train)
knn

knn_train_pred = knn.predict(x_train)

knn_test_pred = knn.predict(x_test)

print(knn_train_pred,knn_test_pred)#result 

Outcome = pd.DataFrame(columns=['Model','Train R2','Test RMSE','Variance'])

r2 = r2_score(y_test,knn_test_pred)
r2_train = r2_score(y_train,knn_train_pred)
rmse = np.sqrt(mean_squared_error(y_test,knn_test_pred))
variance = r2_train - r2
Outcome = Outcome.append({'Model':'K-Nearest Neigbors','Train R2':r2_train,'Test R2':r2,'Test RMSE':r2,'Variance':variance},ignore_index=True)
print('R2:',r2)
print('RMSE:',rmse)

Outcome.head()

svr = SVR().fit(x_train,y_train)
svr

#we use this for train and test prediction
svr_train_pred = svr.predict(x_train)
svr_test_pred = svr.predict(x_test)

print(svr_train_pred,svr_test_pred)

r2 = r2_score(y_test,svr_test_pred)
r2_train = r2_score(y_train,svr_train_pred)
rmse = np.sqrt(mean_squared_error(y_test,svr_test_pred))
variance = r2_train - r2
Outcome = Outcome.append({'Model':'K-Nearest Neigbors','Train R2':r2_train,'Test R2':r2,'Test RMSE':r2,'Variance':variance},ignore_index=True)
print('R2:',r2)
print('RMSE:',rmse)

Outcome.head()

import statsmodels.api as sm

x_train_constant = sm.add_constant(x_train)

model = sm.OLS(y_train,x_train_constant).fit()

model.params

print(model.summary())

plt.scatter(x_train,y_train)
plt.plot(x_train,6.9955 + 0.0541 * x_train, 'y')
plt.show()

y_train_pred = model.predict(x_train_constant)
result = (y_train - y_train_pred)
result

y_train_pred

fig = plt.figure()
sns.distplot(result,bins = 15)
fig.suptitle('Error Term', fontsize = 15)
plt.xlabel('Difference in y_train and y_train_pred', fontsize = 15)
plt.show()

plt.scatter(x_train,result)
plt.show()

x_test_constant = sm.add_constant(x_test)
y_pred = model.predict(x_test_constant)

y_pred

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

np.sqrt(mean_squared_error(y_test,y_pred))

r2 = r2_score(y_test,y_pred)
r2

plt.scatter(x_test,y_test)
plt.plot(x_test, 6.995 + 0.0541 * x_test, 'y')
plt.show()