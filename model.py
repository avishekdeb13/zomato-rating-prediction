# Necessary libraries
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import  r2_score

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

import pickle


# Datafile load
datafile = pd.read_csv("zomato.csv")

# Datafile Exploration
df = datafile.copy()
df.head(3)
df.info()
df.describe()

## EDA
df.isnull().sum()
# Dropping of the null values
df.dropna(how='any',inplace=True)

df.columns
# Dropping of unnecessary columns
df = df.drop(['url', 'address', 'phone','reviews_list' ], axis = 1)

# Renaming columns
df = df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',
                                  'listed_in(city)':'city'})
df.columns

# Modifying the cost column values
df['cost'].unique()
df['cost'] = df['cost'].apply(lambda x: x.replace(',',''))
df['cost'] = df['cost'].astype(int)

# Modifying the Rate column
df['rate'].unique()
df = df.loc[df.rate !='NEW']
df['rate'] = df['rate'].apply(lambda x: x.replace('/5',''))

## DATA VISUALISATION

# Famous Resturants In Ascending order
plt.figure(figsize=(15,10))
chains=df['name'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index)
plt.title("Most famous restaurants chains in Bangaluru")
plt.xlabel("Number of outlets")
plt.show()

# Table booking facilities allowed or not
plt.title("Table booking facilities allowed or not")
sns.countplot(df.book_table)

# Online Delivery options avaialable or not
sns.countplot(df['online_order'])
fig = plt.gcf()
fig.set_size_inches(15,5)
plt.title('Whether Restaurants deliver online or Not')
plt.show()

# Rating distribution
plt.figure(figsize=(9,7))
plt.title("Rating distribution")
sns.distplot(df['rate'],bins=20)
plt.show()

# Most popular service type
sns.countplot(df['type']).set_xticklabels(sns.countplot(df['type']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.title('Type of Service')

# Cost Distribution
plt.figure(figsize=(8,8))
plt.title("Cost Distribution")
sns.distplot(df['cost'])
plt.show()

# Most liked dishes
df.index=range(df.shape[0])
likes=[]
for i in range(df.shape[0]):
    array_split=re.split(',',df['dish_liked'][i])
    for item in array_split:
        likes.append(item)
df.index=range(df.shape[0])
print("Count of Most liked dishes in Bangalore")
favourite_food = pd.Series(likes).value_counts()
favourite_food.head(30)

ax = favourite_food.nlargest(n=20, keep='first').plot(kind='bar',figsize=(18,10),title = 'Top 30 Favourite Food counts ')
for i in ax.patches:
    ax.annotate(str(i.get_height()), (i.get_x() * 1.005, i.get_height() * 1.005))

# Famous Resturant types
plt.figure(figsize=(15,7))
rest=df['rest_type'].value_counts()[:20]
sns.barplot(rest,rest.index)
plt.title("Restaurant types")
plt.xlabel("count")


## MODEL PREPARATION

# Conversion of categorical variables into binary
df.online_order[df.online_order == 'Yes'] = 1
df.online_order[df.online_order == 'No'] = 0

df.book_table[df.book_table == 'Yes'] = 1
df.book_table[df.book_table == 'No'] = 0

le = LabelEncoder()
df.location = le.fit_transform(df.location)
df.rest_type = le.fit_transform(df.rest_type)
df.cuisines = le.fit_transform(df.cuisines)
df.menu_item = le.fit_transform(df.menu_item)

# Dropping the columns containing string values
df = df.drop(["name", "dish_liked", "cuisines", "city"], axis = 1)
df = df.drop("type", axis = 1)

# Dependent and independent variable division
y = df['rate']
x =df.drop(columns = ['rate'])

# Dataset Train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

## 1. LINEAR REGRESSION
model_lr = LinearRegression()
model_lr.fit(x_train, y_train)

y_pred = model_lr.predict(x_test)
print(r2_score(y_test, y_pred))

print(f"The model prediction on train dataset: {round(model_lr.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_lr.score(x_test, y_test),2)}")

## 2.SUPPORT VECTOR MACHINE
model_svr = SVR()
model_svr.fit(x_train , y_train)
print(f"The model prediction on train dataset: {round(model_svr.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_svr.score(x_test, y_test),2)}")

## 3.NAIVE BAYES
model_nbg = GaussianNB()
model_nbg.fit(x_train, y_train)
print(f"The model prediction on train dataset: {round(model_nbg.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_nbg.score(x_test, y_test),2)}")

model_nbb = BernoulliNB()
model_nbb.fit(x_train, y_train)
print(f"The model prediction on train dataset: {round(model_nbb.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_nbb.score(x_test, y_test),2)}")

## 4.DECISION TREE
model_dt = DecisionTreeRegressor()
model_dt.fit(x_train, y_train)
print(f"The model prediction on train dataset: {round(model_dt.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_dt.score(x_test, y_test),2)}")

## 5.RANDOM FOREST REGRESSOR
model_rf = RandomForestRegressor()
model_rf.fit(x_train, y_train)
print(f"The model prediction on train dataset: {round(model_rf.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_rf.score(x_test, y_test),2)}")

## 6.EXTRA TREE REGRESSOR
model_et = ExtraTreesRegressor()
model_et.fit(x_train, y_train)
print(f"The model prediction on train dataset: {round(model_et.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_et.score(x_test, y_test),2)}")

## 7.ADAPTIVE BOOSTING
model_ada = AdaBoostRegressor(base_estimator = model_dt)
model_ada.fit(x_train, y_train)
print(f"The model prediction on train dataset: {round(model_ada.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_ada.score(x_test, y_test),2)}")

## 8.GRADIENT BOOSTING
model_gb = GradientBoostingRegressor()
model_gb.fit(x_train, y_train)
print(f"The model prediction on train dataset: {round(model_gb.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_gb.score(x_test, y_test),2)}")


## BEST MODEL SAVE
pickle.dump(model_ada, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))