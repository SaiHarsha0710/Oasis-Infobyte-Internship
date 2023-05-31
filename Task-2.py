import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
data = pd.read_csv("/content/Unemployment in India.csv")
print(data.head())
print(data.isnull().sum())
data.columns=["Region","Data","Frequency","Estimated Unemployment Rate","Estimated Employed",
              "Estimated Labour Participation Rate","Area"]
#correlation
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12,10))
sns.heatmap(data.corr())
plt.show()
#data visualisation
data.columns=["Region","Data","Frequency","Estimated Unemployment Rate","Estimated Employed",
              "Estimated Labour Participation Rate","Area"]
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Employed",hue="Region",data=data)
plt.show()
plt.figure(figsize=(12,10))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate",hue="Region",data=data)
plt.show()