import numpy as np
import pandas as pd
import time
import pandas as pd
import numpy as np
import os
import random
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
pd.options.display.max_rows = None
pd.set_option('display.max_columns', 500)
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
#np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=np.inf)
from pathlib import Path
pathCurrrent= Path.cwd()
pathMainCodes=Path.cwd().parent
pathCurrrent=str(pathCurrrent).replace("\\", '/')
pathMainCodes=str(pathMainCodes).replace("\\", '/')
pathData=pathMainCodes+"/data/paperMachine/"
import time
os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
np.random.seed(42)




from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1,1,1,1,1], [1, 1, 1, 0,0,0,1,1]).ravel()






from tensorflow.keras.utils import to_categorical
a=to_categorical([0, 1, 2])
print("a",a)
b=to_categorical([0, 1, 2], num_classes=3)
print("b",b)
from random import randint

a=1

def c(a=1,b=2,c=3):
    return a,b,c

c(a=a)



i=0
while i in range(8):
    print(i)
    i+=1
    if i==3:
        i+=2
        print(i)



print('before for')

for i in range(8):
    if i==3:
        continue
        #print(3333)
    if i==5:
        #print(4444)
        continue
    print(i)

print('after for');print('maji');print('maji');print('maji');


print('maji')

# manual nested cross-validation for random forest on a classification dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
# cv_inner.split()
# # define the model
# model = RandomForestClassifier(random_state=1)
# # define search space
# space = dict()
# space['n_estimators'] = [10, 100, 500]
# space['max_features'] = [2, 4, 6]
# # define search
# search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
# # execute search
# result = search.fit(X_train, y_train)

a=np.array([[1, 2,5,6], [1, 4,7,8],[2, 6,12,1],[1, 2,5,6], [1, 4,7,8],[2, 6,12,1], [20, 8,9,10],[2, 6,12,1], [20, 8,9,10],[2, 6,12,1]])

#(a[:, 1:] - a[:, 1:].min(axis=0)) / (a[:, 1:].max(axis=0) - a[:, 1:].min(axis=0))

min_max_scaler = MinMaxScaler(feature_range=(0, 1))
x_scaledTrain = min_max_scaler.fit_transform(a[:5, 1:])
x_scaledTest= min_max_scaler.fit_transform(a[5:, 1:])
#b=np.ma.array(a)
#a[[3,5,7]]= np.ma.masked
#a.compressed()
foo = ["a", "b", "c", "d", "e"]

print(foo[randint(0,4)])

l=[0]
f=np.full([2,3],0)
aa=np.array([1])
a=np.array([[0]])
print("hi")

#tr,te=train_test_split(df.values,shuffle=True, test_size=.2, random_state=42,stratify=df.values[:,0])

df= pd.DataFrame([[0, 2, 3,4], [0, 5, 6,5], [1, 8, 9,7]], columns=['a','b','c','d'], index=['q','w','r'])#index=pd.Series(['a', 'b', 'c']
                     #,[1, 8, 9,7],
                  #[1, 8, 9,7],[0, 5, 6,5], [1, 8, 9,7],[1, 8, 9,7],[1, 8, 9,7]])#,columns=['a','b','c','d'])#
#df1=df.drop(['c'], axis=1, inplace=True)
#print(df1)
yX=df.values
jitters=np.empty([0,df.shape[1]])
jitters=np.append(jitters,yX,axis=0)

X = df.iloc[:, 1:].values  # converts the df to a numpy array
y = df.iloc[:, 0].values

skf = StratifiedKFold(n_splits=5,shuffle=True)

for i, (trainIndex, testIndex) in enumerate(skf.split(X,y)):
    print(f"Running Fold {i} -------------")
    #print("TRAIN:", trainIndex, "TEST:", testIndex)

    yXtrain, yXtest = yX[trainIndex], yX[testIndex]

a=np.full([2,4],0)
b=np.empty([0, 4])
#qq=np.array([[0,0,0]])
#qq=np.append(qq,np.array([[1,2,3],[1,2,3],[1,2,3]]),axis=0)# if axis define shapes must be equal, if axis does not define then will be vector
df=pd.DataFrame(columns=['A', 'B', 'C'],index=range(2))
np.append()
for i in range(2):
    for j in range(3):
        a[i,j]=i
        df.iloc[i,:]=i
a[0,0:2]=200
print(df)
print(a)


print('Threshold=%.3f, F-Score=%.5f' % (.012845, .01234367))
df=pd.DataFrame([[1, 2,5,6], [1, 4,7,8], [2, 2,9,10],[2, 2,5,6], [0, 4,7,8], [1, 2,9,10]
                 ,[1, 2,5,6], [1, 4,7,8], [2, 2,9,10],[2, 2,5,6], [0, 4,7,8], [1, 2,9,10]])
# X = np.array([[1, 2,5,6], [3, 4,7,8], [1, 2,9,10], [3, 4,11,12]])
# y = np.array([1, 2, 3, 4])

X=df.iloc[:,1:].values
y=df.iloc[:,0].values
#y=to_categorical(y)
skf = StratifiedKFold(n_splits=5,shuffle=True)
X_train, X_test =[],[]
y_train, y_test =[],[]
for train_index, test_index in skf.split(X,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    print(X[train_index])
    print( X[test_index])
    print(to_categorical(y[train_index]))
    print(to_categorical(y[test_index]))


def x(a,b,c,d=0,*k,**k1): #*k just accept int index k[index],,, **k1 just accept string(dict) index k1[index]
    print(k[0])
    #print(k1['r'])
    print(d)

x(1,2,3,4,7,9)

f1 = {'r': 1.00}
f2=dict(r=77)
x(1,2,3,4,7,**f1)
x(1,2,3,4,7,r=f2['r'])

f3 = {"axis": 0}
f4=dict(axis=0)
print(np.concatenate([[1,2,3],[4,5,6]],**f3))
print(np.concatenate([[1,2,3],[4,5,6]],axis=f4['axis']))



print(np.random.rand(8))
print('\n')
print(random.randrange(8))
print('j')
# dfpShifted5_ForAug=pd.DataFrame()
# l=[]
# dfpShifted5 = pd.DataFrame([[1, 2, 3], [1, 5, 6], [0, 8, 9],[0, 8, 9],[0, 8, 9]],columns=['a','b','c'])
# print(dfpShifted5)
# #dfpShifted5.loc[((dfpShifted5.index >= i) & (dfpShifted5.index<i+5)) & (dfpShifted5.label == 1), :]
# count1=dfpShifted5.iloc[0:0+5,0].values.tolist().count(0)
# a=[3*dfpShifted5.iloc[0,:].tolist()]#duplicate s as length number of zerozes
# b=np.reshape(np.array(dfpShifted5[dfpShifted5['a']==1][0:0+5]),
#              (-1,np.array(dfpShifted5[dfpShifted5['a']==1][0:0+5]).shape[0]*
#               np.array(dfpShifted5[dfpShifted5['a']==1][0:0+5]).shape[1]))# append ones to end of duplicated zeros(s)
#
# c=np.concatenate((a,b),axis=1)# make one row
# dfpShifted5_ForAug = dfpShifted5_ForAug.append(pd.DataFrame(c))
#
# print('c',pd.DataFrame(c))
# print('count1=',count1)
# # print('zeros',a)
# # print('ones',b)
#
# print('dfpShifted5_ForAug',dfpShifted5_ForAug)
# print(dfpShifted5_ForAug.shape)
# dfpShifted5First = pd.read_csv(pathData+"dfpShifted5_ForAug_201201_163829_AllTested_Correct.csv")
# dfpShifted5Second = pd.read_csv(pathData+"dfpShifted5_ForAug_201201_174555.csv")
#
# diff=pd.concat([dfpShifted5First,dfpShifted5Second]).drop_duplicates(keep=False)



import numpy as np
from sklearn.model_selection import train_test_split
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8],[9, 10], [11, 12], [13, 14]])
y = np.array([[0, 0, 1, 1,0, 1, 0]])

df=pd.DataFrame(np.concatenate([X,y.T],axis=1))
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,-1], test_size=0.2,shuffle=True, random_state=42,stratify=df.iloc[:,-1])
print('hi')
# print(diff.values)
# df = pd.DataFrame([[1, 2, 3], [1, 5, 6], [0, 8, 9],[0, 3, 9]],columns=['a','b','c'])
# a=df.astype(float)
# b=np.array([[1, 2, 3], [1, 5, 6]])
# print(a)
#
# print(df)
# x,y=dfpShifted5.iloc[:,1:],dfpShifted5.iloc[:,0]
# train, test = train_test_split(dfpShifted5,test_size=0.30, random_state=42,shuffle=True,stratify=y)
# print(f'train -  {train}\ntest -  {test}')
#
#
# skf = StratifiedKFold(n_splits=5)
# x,y=dfpShifted5.iloc[:,1:],dfpShifted5.iloc[:,0]

# for index in skf.split(x,y):
#         X = x[index]
       # y = y[index]

# for train_index, test_index in kf.split(x,y):
#         train_X, test_X = x[train_index], x[test_index]
#         train_y, test_y = y[train_index], y[test_index]

#
# import numpy as np
# X, y = np.ones((50, 1)), np.hstack(([0] * 45, [1] * 5))
# print(X)
# print('y',y)
# print(dfpShifted5)
# skf = StratifiedKFold(n_splits=2)

# for train, test in skf.split(x, y):
#     #print('train -  {}   |   test -  {}'.format(
#       #  np.bincount(y[train]), np.bincount(y[test])))
#     print(f'train -  {train}   |   test -  {test}')

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

a=np.array([[1,2,3,4],[0,6,3,4]])
# b=np.array([[5,6,7,8]])
#
# aa = a.reshape((-1, a.shape[0], a.shape[1]))
# bb = b.reshape((-1, b.shape[0], b.shape[1]))
#
# print(jitter(np.array([[1,2,3,4],[5,6,7,8]])))
# print(jitter(np.array([5,6,7,8])))
#
# print(scaling(aa))
# print(scaling(bb))
df2=pd.DataFrame()
df=pd.DataFrame(a,columns=None)
df = df.drop([0], axis=1)
df = df.drop([0], axis=0)
#df[1]=df[1].astype(float)
print(df)
# for i in range(len(df)):
#     if df.iloc[i,0]==1 and i==0:#and i==0:
#         df2 = df2.append(df.iloc[i])
#     if df.iloc[i,0]==1 and df.iloc[:i,0].tail(1).values==0 and i!=0:
#         df2 = df2.append(df.iloc[:i].tail(1))
#         df2=df2.append(df.iloc[i])
#     if df.iloc[i, 0] == 1 and df.iloc[:i,0].tail(1).values==1 and i != 0:  # and i==0:
#         df2 = df2.append(df.iloc[i])


#print(df2)
#lambda x: x*10 if x<2 else (x**2 if x<4 else x+10)
#print(df.loc[0,0])

# def b():
#     a(1, 2, 3)
# def a(x,y,z=1,zz=4):
#     print(z)
# b()

