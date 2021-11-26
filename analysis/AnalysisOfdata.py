
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as pltly
from plotly.offline import iplot
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly
import pathlib
pd.options.display.max_rows = None
pd.set_option('display.max_columns', 500)
from dateutil.parser import parse
import matplotlib as mpl
from datetime import datetime

#%%

dfa = pd.read_csv("data/Genesis_AnomalyLabels.csv")
dfs = pd.read_csv("data/Genesis_StateMachineLabel.csv")
dfn = pd.read_csv("data/Genesis_normal.csv")
dfl = pd.read_csv("data/Genesis_lineardrive.csv")
dfp = pd.read_csv("data/Genesis_pressure.csv")

#%%

print(f"dfa: {pd.DataFrame(dfa.describe().transpose()).iloc[:,[1,2,3,7]]}")
print(f"dfs: {pd.DataFrame(dfs.describe().transpose()).iloc[:,[1,2,3,7]]}")
print(f"dfn: {pd.DataFrame(dfn.describe().transpose()).iloc[:,[1,2,3,7]]}")
print(f"dfl: {pd.DataFrame(dfl.describe().transpose()).iloc[:,[1,2,3,7]]}")
print(f"dfp: {pd.DataFrame(dfp.describe().transpose()).iloc[:,[1,2,3,7]]}")


#%%

dfa["Timestamp"]=[datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for x in dfa["Timestamp"]]

dfs["Timestamp"]=[datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for x in dfs["Timestamp"]]

#%%

dfn["Timestamp"]=pd.to_datetime(dfn["Timestamp"], unit='ms')
dfl["Timestamp"]=pd.to_datetime(dfl["Timestamp"], unit='ms')
dfp["Timestamp"]=pd.to_datetime(dfp["Timestamp"], unit='ms')

print(dfa["Timestamp"].min(),dfa["Timestamp"].max())
print(dfs["Timestamp"].min(),dfs["Timestamp"].max())
print(dfn["Timestamp"].min(),dfn["Timestamp"].max())
print(dfl["Timestamp"].min(),dfl["Timestamp"].max())
print(dfp["Timestamp"].min(),dfp["Timestamp"].max())

#%%

dfa=dfa.reset_index()
dfs=dfs.reset_index()
dfn=dfn.reset_index()
dfl=dfl.reset_index()
dfp=dfp.reset_index()

dfa.index=dfa["Timestamp"]
dfs.index=dfs["Timestamp"]
dfn.index=dfn["Timestamp"]
dfl.index=dfl["Timestamp"]
dfp.index=dfp["Timestamp"]


#%%

dfa.drop(['Timestamp','index'], axis=1, inplace=True)
dfs.drop(['Timestamp','index'], axis=1, inplace=True)
dfn.drop(['Timestamp','index'], axis=1, inplace=True)
dfl.drop(['Timestamp','index'], axis=1, inplace=True)
dfp.drop(['Timestamp','index'], axis=1, inplace=True)

#%%

print("dfa")
dfa.iloc[0:5,0:6]

#%%

print("dfs")
dfs.iloc[0:5,0:6]


#%%

print("dfn")
dfn.iloc[0:5, np.r_[0:4,8:12]]

#%%

print("dfl")
dfl.iloc[0:5, np.r_[0:4,8:12]]


#%%

print("dfp")
dfp.iloc[0:5, np.r_[0:4,8:12]]


#%%

# # Draw Plot
# def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
#     plt.figure(figsize=(16,5), dpi=dpi)
#     plt.plot(x, y, color='tab:red')
#     plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
#     plt.show()
#
# plot_df(dfn, x=dfn["Timestamp"], y=dfn["MotorData.ActCurrent"], title='MotorData.ActCurrent',)
# # plot_df(dfn, x=dfn["Timestamp"], y=dfn["MotorData.ActPosition"], title='MotorData.ActPosition')
# plot_df(dfn, x=dfn["Timestamp"], y=dfn["MotorData.ActSpeed"], title='MotorData.ActSpeed')
# plot_df(dfn, x=dfn["Timestamp"], y=dfn["MotorData.IsAcceleration"], title='MotorData.IsAcceleration')
# plot_df(dfn, x=dfn["Timestamp"], y=dfn["MotorData.IsForce"], title='MotorData.IsForce')
# plot_df(dfn, x=dfn["Timestamp"], y=dfn["Motor_Pos1reached"], title='Motor_Pos1reached')


#%%

dfa.iloc[:5,1]

#%%

dfa["MotorData.ActCurrent"][:10].plot()

#%%

fig, axr = plt.subplots(1, 1, figsize=(10,10))
plt.plot(dfa.iloc[:100,1])
fig.tight_layout(h_pad=20)
plt.show()