# $ 0 U l $ h ! f T 3 r

# IMPORT LIBRARIES

import numpy as np

import pandas as pd
pd.set_option('display.expand_frame_repr', False)

import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


# DATA LOAD

train_set = pd.read_csv("/home/vic/Desktop/Kaggle/Titanic/train.csv")
target = train_set.Survived
test_set = pd.read_csv("/home/vic/Desktop/Kaggle/Titanic/test.csv")

# DATA CHECK

#print(train_set.shape)
#print(train_set.head())
#print(train_set.describe())

# DATA CHECK

#print(train_set.describe())

# COMBINING TRAIN AND TEST
to_test = train_set.Survived
train_set.drop('Survived',axis=1,inplace=True)
combined = train_set.append(test_set)
combined.reset_index(inplace=True)
combined.drop('index',axis=1,inplace=True)

#print(combined.head())
#print(combined.shape)

# VISUALIZATIONS OF DATA BASED ON DIFFERENT FEATURES

# VISUALIZATION USING GENDER

"""survived_sex = train_set[train_set['Survived']==1]['Sex'].value_counts()
dead_sex = train_set[train_set['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True,figsize=(15,8))
plt.show()"""

# VISUALIZATION USING AGE

"""figure = plt.figure(figsize=(15,8))
plt.hist([train_set[train_set['Survived']==1]['Age'],train_set[train_set['Survived']==0]['Age']],stacked=True,color=['g','r'],bins=50,label=['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('No.of Passengers')
plt.legend()
plt.show()"""

# VISUALIZATION USING FARE

"""figure = plt.figure(figsize=(15,8))
plt.hist([train_set[train_set['Survived']==1]['Fare'],train_set[train_set['Survived']==0]['Fare']],stacked=True,color=['g','r'],bins=50,label=['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('No.of Passengers')
plt.legend()
plt.show()"""

# VISUALIZATION USING EMBARKATION SITE

"""survived_em = train_set[train_set['Survived']==1]['Embarked'].value_counts()
dead_em = train_set[train_set['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_em,dead_em])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True,figsize=(15,8))
plt.show()"""


# FUNCTION TO FILL AGE ACCORDING TO TITLES

def get_titles():
    global combined
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

get_titles()

# print(combined.tail())

combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
combined['Age'] = combined.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x : x.fillna(x.median()))

#print(train.info())

# FILLING REST UNFILLED VALUES

combined['Fare'] = combined.groupby(['Sex','Pclass','Title'])['Fare'].transform(lambda x:x.fillna(x.median()))
combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
combined['Familysize'] = combined['SibSp'] + combined['Parch'] + 1
combined['Single'] = combined['Familysize'].map(lambda s : 1 if s==1 else 0)
combined['Couple'] = combined['Familysize'].map(lambda s : 1 if s==2 else 0)
combined['Family'] = combined['Familysize'].map(lambda s : 1 if 3<=s else 0)
combined.Embarked.fillna('S',inplace=True)
combined.Cabin.fillna('U',inplace=True)
combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])


# CREATING DUMMIES

class_feature = pd.get_dummies(combined['Pclass'],prefix='Pclass')
titles_feature = pd.get_dummies(combined['Title'],prefix='Title')
embarked_feature = pd.get_dummies(combined['Embarked'],prefix='Embarked')
cabin_feature = pd.get_dummies(combined['Cabin'],prefix='Cabin')
combined = pd.concat([combined,class_feature],axis=1)
combined = pd.concat([combined,titles_feature],axis=1)
combined = pd.concat([combined,embarked_feature],axis=1)
combined = pd.concat([combined,cabin_feature],axis=1)

combined.drop('Name',axis=1,inplace=True)
combined.drop('Pclass',axis=1,inplace=True)
combined.drop('Ticket',axis=1,inplace=True)
combined.drop('Title',axis=1,inplace=True)
combined.drop('Cabin',axis=1,inplace=True)
combined.drop('Embarked',axis=1,inplace=True)

#print(combined.info())


# CHECKING IMPORTANT FEATURES

train = combined.ix[0:890]
test = combined.ix[891:1308]
clf = ExtraTreesClassifier(n_estimators=200)
clf.fit(train,to_test)
features = pd.DataFrame()
features['feature'] = train.columns
features['importances'] = clf.feature_importances_
features.sort_values(['importances'],ascending=[False])

# TRAINING

model = SelectFromModel(clf,prefit=True)
training = model.transform(train)
test = test.fillna(method='ffill')
testing = model.transform(test)

forest = RandomForestClassifier(max_features='sqrt')
parameter_grid = {'max_depth':[10],'n_estimators': [250],'criterion': ['gini','entropy']}
cross_validation = StratifiedKFold(n_splits=10)
grid_search = GridSearchCV(forest,param_grid=parameter_grid,cv=cross_validation)
grid_search.fit(training,target)


# CREATING CSV FILE

pipeline = grid_search
output = pipeline.predict(testing).astype(int)
df_op = pd.DataFrame()
df_op['PassengerId'] = test['PassengerId']
df_op['Survived'] = output
df_op[['PassengerId','Survived']].to_csv("/home/vic/Desktop/Kaggle/Titanic/final.csv",index=False)





