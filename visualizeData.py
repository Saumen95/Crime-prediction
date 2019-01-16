'''
##data viz using sns
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

crime = pd.read_csv('crime-data_crime-data.csv')
crime.head()

crime.dropna()
crime.info()

#sns.factorplot("UCR CRIME CATEGORY", data=crime, aspect=2,kind="count", color='steelblue')
#sns.factorplot("PREMISE TYPE", data=crime, aspect=2,kind="count", color='steelblue')

sns.factorplot("OCCURRED ON", data=crime, aspect=2,kind="count", color='steelblue')

plt.show()
'''
##normal data viz
import numpy as np
import matplotlib.pyplot as plt
import csv
INC_NUMBER=[]
OCCURRED_ON=[]
OCCURRED_TO=[]
CRIME_CATEGORY=[]
BLOCK_ADDR=[]
PREMISE_TYPE=[]
with open('testData.csv','r') as csvfile:
    plots=csv.reader(csvfile,delimiter=',')
    for row in plots:
        INC_NUMBER.append(str(row[0]))
        OCCURRED_ON.append(str(row[1]))
        OCCURRED_TO.append(str(row[2]))
        CRIME_CATEGORY.append(str(row[3]))
        BLOCK_ADDR.append(str(row[4]))
        PREMISE_TYPE.append(str(row[6]))

from collections import Counter

crime_year=[]
for i in INC_NUMBER:
    if i in crime_year:
        pass
    else:
        crime_year.append(i[0:4])

by_year=Counter(crime_year[1:])
print(by_year)
plt.bar(range(len(by_year)), list(by_year.values()), align='center')
plt.xticks(range(len(by_year)), list(by_year.keys()))
plt.show()

by_category=Counter(CRIME_CATEGORY[1:])
print(by_category)
plt.bar(range(len(by_category)), list(by_category.values()), align='center')
plt.xticks(range(len(by_category)), list(by_category.keys()))
plt.show()


by_premiseType=Counter(PREMISE_TYPE[1:])
print(by_premiseType)
plt.bar(range(len(by_premiseType)), list(by_premiseType.values()), align='center')
plt.xticks(range(len(by_premiseType)), list(by_premiseType.keys()))
plt.show()