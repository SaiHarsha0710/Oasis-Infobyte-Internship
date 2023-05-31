import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.simplefilter("ignore")
from google.colab import files 
uploaded = files.upload()
import io
df = pd.read_csv(io.BytesIO(uploaded['Iris.csv']))
df
df.info()
df.isnull().sum()
df=df.drop(columns="Id")
df
df1=df.groupby('Species')
df1.head()
df['Species'].value_counts()
df['Species'].unique()
plt.boxplot(df['SepalLengthCm'])
plt.boxplot(df['PetalWidthCm'])
sns.heatmap(df.corr())
x=df.iloc[:,0:4]
x
y=df.iloc[:,4]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
x_train.shape
x_test.shape
y_test.shape
y_train.shape
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
y_pred=model.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}".format(accuracy))