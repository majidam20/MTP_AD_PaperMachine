import os
import sys
import random
import time
import pandas as pd
import numpy as np
pd.options.display.max_rows = None
pd.set_option('display.max_columns', 500)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=np.inf)

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)


from pathlib import Path
pathCurrrent = Path.cwd()
pathMainCodes = Path.cwd().parent
pathCurrrent = str(pathCurrrent).replace("\\", '/')
pathMainCodes = str(pathMainCodes).replace("\\", '/')

pathDataforShifted5 = pathMainCodes + "/data/paperMachine/forCV/forShifted5/"
pathDataRolLbl1 = pathMainCodes + "/data/paperMachine/forCV/Shifted5_Rol_Lbl1/"

pathSavingPlotsShifted5=pathMainCodes + "/reports/forShifted5/"

shiftedNumber=5

1#start curve_shift
###***start curve_shift ##########################################################
# sign = lambda x: (1, -1)[x < 0]
# def curve_shift(df, shift_by):
#     '''
#     This function will shift the binary labels in a dataframe.
#     The curve shift will be with respect to the 1s.
#     For example, if shift is -2, the following process
#     will happen: if row n is labeled as 1, then
#     - Make row (n+shift_by):(n+shift_by-1) = 1.
#     - Remove row n.
#     i.e. the labels will be shifted up to 2 rows up.
#
#     Inputs:
#     df       A pandas dataframe with a binary labeled column.
#              This labeled column should be named as 'label'.
#     shift_by An integer denoting the number of rows to shift.
#
#     Output
#     df       A dataframe with the binary labels shifted by shift.
#     '''
#
#     vector = df['label'].copy()
#     for s in range(abs(shift_by)):
#         tmp = vector.shift(sign(shift_by))
#         tmp = tmp.fillna(0)
#         vector += tmp
#     labelcol = 'label'
#     # Add vector to the df
#     df.insert(loc=0, column=labelcol + 'tmp', value=vector)
#     # Remove the rows with labelcol == 1.
#     df = df.drop(df[df[labelcol] == 1].index)
#     # Drop labelcol and rename the tmp col as labelcol
#     df = df.drop(labelcol, axis=1)
#     df = df.rename(columns={labelcol + 'tmp': labelcol})
#     # Make the labelcol binary
#     df.loc[df[labelcol] > 0, labelcol] = 1
#
#     return df
# # end curve_shift#############################################################

2#Shift data corresponds to lookback length
# ###*** Shift data corresponds to lookback length
# dfpNoLess5 = pd.read_csv(pathDataforShifted5+"dfpAllScale_2RowsDel_NoLess5_210123_160259.csv",header=None)
# #dfpNoLess5=dfpNoLess5[:1000]
#
# #dfpnt = dfpNoLess5.drop(["time"], axis=1, inplace=True)
#
# shiftedNumber=5
# dfpNoLess5.rename(columns={0: 'label'}, inplace=True)
# dfpShifted5NoLess5 = curve_shift(dfpNoLess5, shift_by = -1*shiftedNumber)
# dfpShifted5NoLess5 = dfpShifted5NoLess5.astype({"label": int})
# dfpShifted5NoLess5.rename(columns={'label': 0}, inplace=True)
# dfpShifted5NoLess5=dfpShifted5NoLess5.reset_index(drop=True)
#
# datestr = time.strftime("%y%m%d_%H%M%S")
# dfpShifted5NoLess5.to_csv(pathDataforShifted5+f"dfpShifted{shiftedNumber}_NoLess5_{datestr}.csv",index=None,header=None)
# print(f"Creation dfpShifted_{shiftedNumber} is Done!!!")
#################################################################################


22# Rolling shifted(1,2,3,4,5) DSs

def temporalize(X, y, lookback):
    output_X = []
    output_y = []

    for i in range(-2, len(X) - 5 - 1):
        t = []
        for j in range(1, 5 + 1):
            t.append(X[i + j + 1, :])

        lookback = 4### because i start from zero and 4th(0,1,2,3,4) value is first label for first block
        i+=2
        output_X.append(t)
        output_y.append(y[i + lookback ])
    y2=np.repeat(np.array(output_y),5)
    y2=y2.reshape(y2.shape[0],1)
    X2=np.array(output_X).reshape(len(output_X)*5,59)
    y2X2=np.concatenate((y2, X2), axis=1)
    return y2X2

dfpShifted5 = pd.read_csv(pathDataforShifted5+"dfpShifted5_NoLess5_210216_182303.csv",header=None)

input_X = dfpShifted5.iloc[:, 1:].values  # converts the df to a numpy array
input_y = dfpShifted5.iloc[:,0].values

lookback = 5  # Equivalent to 10 min of past data.

# Temporalize the data
yX = temporalize(X = input_X, y = input_y, lookback = lookback)
dfpShifted5Rolling=pd.DataFrame(yX)

datestr = time.strftime("%y%m%d_%H%M%S")
dfpShifted5Rolling.to_csv(pathDataforShifted5+f"dfpShifted{shiftedNumber}_Rolling_{datestr}.csv",index=None,header=None)
print(f"Creation dfpShifted{shiftedNumber}_Rolling_{datestr}.csv is Done!!!")