import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import LocalOutlierFactor

import pandas as pd

path = r'C:\my-data'
df = pd.read_csv(path + '/clustering_data.csv')
filt = (df.X > -1.71) & (df.X < 10.51)

df_split = pd.pivot(df[filt],values = 'Z-RAW',index = ['X','Y'], columns = 'UNIQUE_ID').dropna()

all_ids = list(df_split.keys())
all_hds = [w[:10] for w in all_ids]
hds = np.unique(all_hds)
pred_dict = {}
pred_dict['LOF3'] = []
pred_dict['LOF10'] = []
pred_dict['UNIQUE_ID'] = []

for hd in hds:
    X1 = []
    for hd_id in all_ids:
        if hd_id[:10] == hd:
            X1.append(list(df_split[hd_id]))
            pred_dict['UNIQUE_ID'].append(hd_id)
        X = np.array(X1)
    for nn in [3,10]:
        lab = 'LOF'+str(nn)
        algorithm = LocalOutlierFactor(n_neighbors==nn,novelty=False)
        algorithm.fit(X)
        y_pred = algorithm.fit_predict(X)
        
        pred_dict[lab].extend(list(y_pred))
        
df_out = pd.DataFrame(pred_dict)
df_out.to_csv(path+'/anomaly_predict.csv')
