import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#normalize function
def min_max_norm(df, columns):
    for column in columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df
  
  # kde_val_nonnorm = pd.read_csv('/Users/stephaniewanko/Downloads/water_tracking/non_norm/135l_KDE_nonnorm.csv')
kde_band = pd.read_csv('/Users/stephaniewanko/Downloads/water_tracking/lower_bandwidth/135l_KDE_water_lowerbandwidth.csv')
ML_band = pd.read_csv('/Users/stephaniewanko/Downloads/water_tracking/lower_bandwidth/likelihood_135l_lower_band.csv')
kde_water = pd.read_csv('/Users/stephaniewanko/Downloads/water_tracking/water_norm/135l_KDE_water_water_norm.csv')
ML_water = pd.read_csv('/Users/stephaniewanko/Downloads/water_tracking/water_norm/likelihood_135l_water_norm.csv')
kde_pro = pd.read_csv('/Users/stephaniewanko/Downloads/water_tracking/protein_norm/135l_KDE_water_protein_norm.csv')
ML_pro = pd.read_csv('/Users/stephaniewanko/Downloads/water_tracking/protein_norm/likelihood_135l_protein_norm.csv')

kde_none = pd.read_csv('/Users/stephaniewanko/Downloads/water_tracking/non_norm/135l_KDE_nonnorm.csv')
#ML_non = pd.read_csv('/Users/stephaniewanko/Downloads/water_tracking/non_norm/likelihood_135l_protein_norm.csv')

kde_all_water = pd.read_csv('/Users/stephaniewanko/Downloads/water_tracking/water_norm/135l_all_protein_water_norm.csv')
kde_all = pd.read_csv('/Users/stephaniewanko/Downloads/water_tracking/non_norm/135l_all_protein_no_norm.csv')

EDIA = pd.read_csv('/Users/stephaniewanko/Downloads/water_tracking/EDIA_values_135l.csv')
EDIA = EDIA.sort_values(by=['Substructure id'])

#EDIA_norm = min_max_norm(EDIA, ['EDIA', 'B factor'])
kde_band_norm = min_max_norm(kde_band, ['bfactor'])
ML_band_norm = min_max_norm(ML_band, ['likelihood'])

kde_water_norm = min_max_norm(kde_water, ['bfactor'])
ML_water_norm = min_max_norm(ML_water, ['likelihood'])

kde_pro_norm = min_max_norm(kde_pro, ['bfactor'])
ML_pro_norm = min_max_norm(ML_pro, ['likelihood'])

kde_none_norm = min_max_norm(kde_none, ['bfactor'])
ML_none_norm = min_max_norm(ML_band, ['likelihood'])

kde_all_water_norm = min_max_norm(kde_all_water, ['bfactor'])
kde_all_norm = min_max_norm(kde_all, ['bfactor'])


fig = plt.figure()
ax = plt.subplot(111)
plt.plot(EDIA_norm['Substructure id'], EDIA_norm['EDIA'], label='EDIA')
#plt.plot(EDIA_norm['Substructure id'], EDIA_norm['B factor'], label='B-Factor')
#plt.plot(kde_band_norm['resid'], kde_val_band_norm['bfactor'], label='Band 0.5 KDE')
#plt.plot(kde_water_norm['resid'], kde_water_norm['bfactor'], label='Water Normalized')
#plt.plot(kde_pro_norm['resid'], kde_pro_norm['bfactor'], label='Protein Normalized')
#plt.plot(kde_none_norm['resid'], kde_none_norm['bfactor'], label='No Norm')
#plt.plot(ML_none_norm['resid'], ML_none_norm['likelihood'], label='ML No Norm')
plt.plot(kde_all_water_norm['resid'], kde_all_water_norm['bfactor'], label='All Protein Water Normalized')
plt.plot(kde_all_norm['resid'], kde_all_norm['bfactor'], label='All Protein')
plt.xlabel('Water Residue Number')
plt.ylabel('Relative value (min-max normalized)')
ax.legend(bbox_to_anchor=(1.6, 1.05))

#plt.plot(ML_band_norm['resid'], ML_band_norm['likelihood'], label='Band 0.5 ML')

EDIA_all = EDIA.merge(kde_all_water_norm, left_on='Substructure id', right_on='resid')
EDIA_all['difference'] = EDIA_all['EDIA'] - EDIA_all['bfactor']
EDIA_all.sort_values(by=['difference'])
