import numpy as np
import pandas as pd
from pathlib import Path
pd.options.display.max_rows = None
pd.set_option('display.max_columns', 500)
#import pathlib
p = Path.cwd()
# p5=os.path.normpath(p5)
# p5.replace(os.sep, '/')
filename=str(p.parent)+"\\"+"AtrialFibrillation"+"\\"+"AtrialFibrillation_TRAIN"#"DiatomSizeReduction_TRAIN"#"dfShifted2NoR_TEST"#dfShifted2NoR,DiatomSizeReduction_TEST
#df=np.loadtxt(filename, delimiter = ',')#,dtype=np.float
df=np.genfromtxt(filename, delimiter=',')#[:,:]
# #df=pd.read_csv(filename)
#print(df)#',)
from sklearn.model_selection import train_test_split
# dfShifted2=pd.read_csv("dfShifted2.csv")
# print(dfShifted2.values)
#train, test = train_test_split(dfShifted2, test_size=0.3, random_state=42,shuffle=False)
#print(test)
df=pd.DataFrame(df)
print(pd.DataFrame(df.describe().transpose()).iloc[:,[1,2,3,7]])

import shutil
shutil.rmtree('/path/to/folder')



