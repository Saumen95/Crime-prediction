import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
from sklearn.linear_model import LinearRegression
matplotlib inline


crimedata = pd.read_csv(/home/minix/project 4-2/crime-data_crime-data.csv)
crimedata.head()
crimedata.info()
crimedata.describe()
crimedata.columns


sns.pairplot(crimedata)
crimedata.corr()


X = crimedata[OCCURED ON]
y = crimedata[UCR CRIME CATEGORY]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)


predictions = lm.predict(X_test)
pl.scatter(y_test, predictions)
