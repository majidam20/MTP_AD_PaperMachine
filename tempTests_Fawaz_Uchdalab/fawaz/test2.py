import numpy as np
import pandas as pd

l=[]
l2=[]
df=pd.DataFrame()

d=pd.DataFrame([[1,2,3],[4,5,6],[7,8,9],[1,2,3],[4,5,6]
                   ,[7,8,9],[1,2,3],[4,5,6],[7,8,9],[1,2,3]
                   ,[7,8,9],[1,2,3],[4,5,6],[7,8,9],[1,2,3],[1,2,3]
                    ,[7,8,9],[1,2,3],[4,5,6],[7,8,9],[1,2,3],[1,2,3]
,[7,8,9],[1,2,3],[4,5,6],[7,8,9],[1,2,3],[1,2,3],[1,2,3],[4,4,4]

])
print(d)

flag=False
#
for i in range(len(d)):

    if np.mod(i,5)==0 and i>0:
        l=np.reshape(l,(1,np.shape(l)[0]*np.shape(l)[1]))
        df=df.append(pd.DataFrame(l))
        l=[]
        l.append(d.iloc[i, :])
    else:
        l.append(d.iloc[i,:])


print(df)

