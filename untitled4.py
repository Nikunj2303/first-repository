# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 01:20:31 2023

@author: Admin
"""
import numpy as np

import pandas as pd

df=pd.read_csv ('/Users/Admin/OneDrive/Desktop/anconda/python/nik.csv')

print(df.columns)

print(df.dtypes)

print(df['State'].dtypes)


df.Y2008 ==df.Y2008.astype(float)
print( 'Y2008',df.Y2008.dtypes)
print(df.dtypes)

print(df.shape)

print(df.head())
print(df.head(2))

print(df[0:5])
print(df.iloc[0:5])

ser=pd.Series([1,2,3,1,2],dtype="category")
print(ser)

print(df.Index.unique())
print(pd.crosstab(df.Index,df.State))

print(df.sample(n=5))
print(df.sample(frac=0.1))

print(df['State'])
print(df.State)

print(df[["Index","State","Y2008"]])
print(df.loc[:,["Index","State","Y2008"]])

print(df.loc[0:2,["Index","State","Y2008"]])
print(df.loc[:,"Index":"Y2008"])
print(df.iloc[:,0:5])

x = pd.DataFram({"var1": np.arange(1,20,2)},
                index=[9,8,7,6,10, 1, 2, 3, 4, 5])
print(x)
print(x.iloc[:3])
print(x.loc[:10])

data= pd.DataFrame({"A" :["john","Mary","Julia","kenny","Henry"],
                   "B" :["Libra","Capricon","Aries","Scorpio","Aquarius"]})
print(data)
data.colum = ['Name','Zodiac signs']
print('\n',data)

data.columns = ['Name','Zodiac signs']
print(data)

data.rename(columns ={"Names":"Cust_Name"},inplace = True)
print('\n',data)

df.colums = df.colums.str.replace('Y' , 'Year')
print(df.colums)
df.set_index("Index",inplace = True)
print(df.head())
print(df.colums,'\n')
print(df.reset_index(inplace=True))
print('\n',df.head())

print(df.drop('Index',axis = 1))

print(df.drop('Index',axis = "columns"))

print(df.drop(['Index','State'],axis=1))

print(df.drop(0,axis = 0))

print(df.drop(0,axis = "index"))

print(df.drop([0,1,2,3],axis = 0))

print(df.sort_values("State",ascending = False))

print(df.Y2006.sort_values())

print(df.sort_values(["Index","Y2002"]))

df["differece"] = df.Y2008-df.Y2009
print(df)
df["difference"] = df.Y2008-df.Y2009

df["difference2"] = df.eval("Y2008 - Y2009")
print(df.head())
print(df.ratio == df.Y2008/df.Y2009)

print(df.describe())
print(df.describe(include = ['object']))
print(df.mean())

print(df.median())
print(df.agg(["mean","median"]))
print(df.Y2008.mean())
print(df.Y2008.median())
print(df.Y2008.min())
print(df.loc[:,["Y2002","Y2008"]].max())

print(df.groupby("Index")["Y2002","Y2003"].min())
print(df.groupby("Index")["Y2002","Y2003"].agg(["min","max","mean"]))
print(df.groupby("Index").agg({"Y2002": ["min","max"],"Y2003" : "mean"}))

print(df.group("Index").agg({"Y2002" : [("Y2002_min","min"),("Y2002_max","max")],
                             "Y2003" : [("Y2003_mean","mean")]}))
dt = df.groupby("Index").agg({"Y2002": ["min","max"], "Y2003" : "mean"})
dt.colums = ['Y2002_min', 'Y2002_max','Y2003_mean']
print(dt)

print(df.groupby(["Index", "State"]).agg({"Y2002": ["min","max"],"Y2003" : "mean"}))
dt = df.groupby(["Indrx","State"],as_index=False)["Y2002","Y2003"].min()
print(df[df.Index == "A"])
print(df.loc[df.Index == "A",:])
                        
print(df.loc[df.Index == "A","Stare"])
print(df.loc[(df.Index == "A") & (df.Y2002 > 150000),:])
print(df.loc[(df.Index == "A" ) | (df.Index == "W" ),:])
print(df.loc[df.Index.isin(["A","W"]),:])
print(df.query('Y2002 > 1700000 & Y2003 > 1500000'))
       
mydata = {'Crop': ['Rice', 'Wheat', 'Barley', 'Maize'],
          'Yield': [1010, 1025.2, 1404.2, 1251.7],
          'cost' : [102, np.nan, 20, 68] }
crops = pd.DataFram(mydata)
print(crops)
print(crops.isnull())
print(crops.notnull())
print(crops.isnull().sum())

print(crops[crops.cost.isnull()])
print(crops[crops.cost.isnull()].Crop )
print(crops[crops.cost.notnull()].Crop )
print(crops.dropna(how = "any").shape)
print(crops.dropna(how = "all").shap)
print(crops.dropna(subset = ['Yield',"cost"],how ='any').shap)
print(crops.dropna(subset = ['Yield',"cost"],how ='all').shap)
crops['cost'].fillna(value = "UNKNOWN",inplace = True)
print(crops)

data = pd.DataFrame({"Items" : ["TV","Washing Machine","Mobile","TV","TV","Washing Machine"],
                     "Price" : [10000,50000,20000,10000,10000,40000]})
print(data.loc[data.duplicated(),:])
print(data.loc[data.duplicated(keep = "first"),:])
print(data.loc[data.duplicated(keep = "last"),:])
print(data.loc[data.duplicated(keep = False),:])

print(data.drop_duplicates(keep = "first"))
print(data.drop_duplicates(keep = "last"))
print(data.drop_duplicates(keep = False,inplace = True))
iris = pd.read_csv("'/Users/Admin/OneDrive/Desktop/anconda/python/nik.csv'")
print(iris)
iris["setosa"] = iris.species.map({"setosa" : 1,"versicolor":0, "virginica": 0})
print(iris.head())

print(pd.get_dummies(iris.species,prefix = "species"))
print(pd.get_dummies(iris.species,prefix = "species").iloc[:,0:1])
species_dummies = pd.get_dummies(iris.species,prefix = "species").iloc[:,0]
print(species_dummies)
iris = pd.concat([iris,species_dummies],axis = 1)
print(iris.head())

print(pd.get_dummies(iris,columns = ["soecies"],drop_first = True).head())
print(iris.rank())
iris['Rank2'] = iris['sepal_length'].groupby(iris["species"]).rank(ascending=1)
print(iris.head())

iris["cum_sum"] = iris["sepal_length"].cumsum()
print(iris.head())
print(iris.quantile(0.5))

print(iris.quantile([0.1,0.2,0.5]))
print(iris.quantile(0.55))
students =pd.DataFrame({'Name': ['John','Mary','Henry','Augustus','kenny'],
                       'Zodiac Signs': ['Aquarius','Libra','Gemini','Pisces','Virgo']})
def name(row):
    if row["Names"] in ["john","Henry"]:
        return "yes"
    else:
        return "no"
students['flag'] = students.apply(name, axis=1)
print(students) 

students['flag'] = np.where(students['Names'].isin(['John','Henry']),'yes','no')
print(students)
def mname(row):
    if row["Names"] == "John" and row["zodiac signs"] == "Aquarius" : 
        return "yellow"
    elif row["Name"] == "Mary"and row["zodiac signs"] == "Libra" :
        return "blue"
    elif row["zodiac signs"] == "Pisces" :
        return "blue"
    else:
        return "black"
students['color'] = students.apply(mname, axis=1)
print(students)
conditions = [
    (students['Name'] == 'John') & (students['Zodiac signs'] == 'Aquarius'),
    (students['Name'] == 'Mary') & (students['Zodiac signs'] == 'Libra'),
    (students['Zodiac Signs'] == 'purple')]
choices = ['yellow', 'blue', 'purple']
students['color'] = np.select(conditions, choices, default= 'black')
print(students)

data1 = iris.select_dtypes(include=[np.number])
print(data.head())
data3 = iris._get_numeric_data()
print(data3.head(3))    
data4 = iris.select_dtypes(includ = ['object'])
print(data4.head(2))
students = pd.DataFrame({'Name': ['John','Mary','Augustus', 'Kenny'],
                         'Zodiac signs': ['Aquarius','Libra','Gemini','Pisces','Virgo']}) 
students2 = pd.DataFrame({'name': ['John','Mary','Henry','Augustus','Kenny'],
                          'Marks' : [50,81,98,25,35]}) 
data = pd.concat([students,students2])
print(data)

data = pd.concat([students,students2],axis = 1)
print(data)
print(students.append[students2])
classes = {'x':students, 'y': students2}
result = pd.concat(classes)
print(result)

result = pd.merge(students, students2, on='Names')
print(result)
result = pd.merge(students, students2,on='Names',how ="outer")
print(result)
result = pd.merge(students, students2, on='Names',how ="left")
print(result)
result = pd.merge(students, students2, on='Names',how = "right",indicator = True)
print(result)                   

