
# coding: utf-8

# In[1]:


# Titanic dataset from https://www.kaggle.com/c/titanic
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import re

start_time = time.time()

df_train = pd.read_csv('../data/titanic/train.csv')
df_test = pd.read_csv('../data/titanic/test.csv')


# In[2]:


# Adapted from https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python/notebook
for c in list(df_train):
    percent_missing = (df_train[c].isnull().sum() / df_train.shape[0]) * 100
    if percent_missing:
        print(f'% of missing "{c}" records: {percent_missing}')


# In[3]:


df_list = [df_train, df_test]
for (i, df) in enumerate(df_list):
    df['Age'].fillna(df['Age'].median(skipna=True), inplace=True)
    df['Fare'].fillna(df['Fare'].median(skipna=True), inplace=True)
    df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)
    df['Cabin'].fillna(df['Cabin'].value_counts().idxmax(), inplace=True)
    df.Cabin = df.Cabin.map(lambda x: re.compile('[a-zA-Z]').search(x).group())
    df_list[i] = pd.get_dummies(df, columns=['Pclass', 'Embarked', 'Sex', 'Cabin'])    
    assert sum(df.isnull().sum() == 0)

df_train, df_test = df_list

for df in df_list:
    for col in ['PassengerId', 'Name', 'Ticket', 'Sex_female']:
        df.drop(col, axis=1, inplace=True)


# In[4]:


from sklearn.feature_selection import RFE

cols = list(df_train)
cols.remove('Survived')

X = df_train[cols]
y = df_train['Survived']

clf = LogisticRegression(solver='liblinear')

# Create RFE model and select 10 features
rfe = RFE(clf, 10)
rfe = rfe.fit(X, y)

# Summarize the selection of the features
print('Selected features: {}'.format(list(X.columns[rfe.support_])))


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
clf.fit(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print('test accuracy:', test_acc)


# In[6]:


print("--- %s seconds ---" % (time.time() - start_time))

