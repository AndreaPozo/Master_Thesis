#!/usr/bin/env python
# coding: utf-8

# In[1]:


# basic
import sys
import os

# common
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import pickle
from mpl_toolkits.basemap import Basemap

#lib
from lib.calibration import *
from lib.predictor_definition_building import *
from lib.PCA_predictor_sst_mld import PCA_EstelaPred_sea_mask, Plot_EOFs_EstelaPred, standardise_predictor_prep
from lib.mda import Normalize, DeNormalize
from lib.plots_kma import plot_3D_kmeans, plot_scatter_kmeans, plot_grid_kmeans
from lib.plots_base import basemap_ibtracs, basemap_var, basemap_scatter, axplot_basemap, basemap_scatter_both
from lib.plots_dwts import colorp, custom_colorp, Plot_DWTs_Mean_Anom, Plot_DWTs_totalmean,Plot_Probs_WT_WT, Plot_Probs_WT_WT_anomaly, Plot_Probs_WT_WT_WT, Plot_DWTs_Probs, Report_Sim_oneyear, Report_Sim, Plot_DWTs_counts, Chrono_dwts_tcs, Chrono_probs_tcs, Plot_dwts_colormap
from lib.plots_tcs import get_storm_color, get_category, Plot_DWTs_tracks
from lib.plots_aux import data_to_discret, colors_dwt
from lib.extract_tcs import Extract_Rectangle, dwt_tcs_count, dwt_tcs_count_tracks

import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image


# <br>
# <br>
# <br>
# 
# #  <font color='navy'>**Statistical Downscaling Method** </font> 

# > [Daily Weather Types (DWTs) classification](#dwt) <br> <br>
# >> [Principal Component Analysis (PCA)](#pca)<br> <br>
#  [K-means clustering](#km)<br> <br>
#  [DWTs plotting](#plot)<br> <br>
#  [Cluster homegeneity](#ch)<br> <br>
#  [DWTs plotting with predictand variables](#plotp)<br> <br>
# >>
#  [DWTs chronology, seasonality and temporal variability](#chrono) <br> <br>
#  [Relationship predictor-predictand](#pp)<br> <br>
# >> [DWTs storm frequency and track counting](#count)<br> <br>
# [Calibration time period prdictand plotting](#cali)<br> <br>
# >>
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# <br />
# <br />
# 
# ## <font color='royalblue'>**Daily Weather Types (DWTs) classification**</font> <a name="dwt"></a>
# 

# **A weather type approach is proposed.**<br>
# **The index predictor is first partioned into a certain number of clusters, DWTS, obtained combining three data mining techniques.**<br>

# <br />
# 
# ### <font color='cornflowerblue'>**Principal Component Analysis (PCA)**</font> <a name="pca"></a>
# 

# **The PCA is employed to reduce the high dimensionality of the original data space and thus simplify the classification process, transforming the predictor fields into spatial and temporal modes.**

# In[2]:


path_p = r'/home/administrador/Documentos/seasonal/seasonal_forecast/new/'
xs = xr.open_dataset(path_p+'xs_index_vars_19822019_2deg_new.nc')
df = pd.read_pickle(path_p+'df_coordinates_pmin_sst_mld_2019.pkl')
xs_trmm = xs_trmm = xr.open_dataset(path_p+'xs_trmm_1982_2019_2deg_new.nc')


# In[3]:


# predictor area
lo_area = [160, 210]
la_area = [-30, 0]


# In[4]:


# PCA parameters
ipca, xds_PCA = PCA_EstelaPred_sea_mask(xs, ['index']) #ipca son las componentes principales, las 416 que salen
xds_PCA


# <br>
# 
# **PCA projects the original data on a new space searching for the maximum variance of the sample data.**<br>
# **The first 237 PCs are captured, which explain the 90 % of the variability as shown:**

# In[5]:


variance_PCA = plot_graph_variance_PCA(xds_PCA)


# <br>
# 
# **The eigenvectors (the empirical orthogonal functions, EOFs) of the data covariance matrix define the vectors of the new space.**<br>
# **They represent the spatial variability.**

# In[41]:


fig_eofs = Plot_EOFs_EstelaPred(xds_PCA, 4);


# <br>
# 
# **The PCs represent the temporal variability.**

# In[7]:


fig_PCA = plot_PCs(xds_PCA,4)


# <br><br>
# 
# ### <font color='cornflowerblue'>**K-means clustering**</font> <a name="km"></a>

# **Daily synoptic patterns of the index predictor are obtained using the K-means clustering algorithm. It divides the data space into 49 clusters, a number that which must be a compromise between an easy handle characterization of the synoptic patterns and the best reproduction of the variability in the data space. Previous works with similar anlaysis confirmed that the selection of this number is adequate <a href="https://www.researchgate.net/profile/Christie-Hegermiller-2/publication/322103268_Multiscale_climate_emulator_of_multimodal_wave_spectra_MUSCLE-spectra/links/5a453d2aaca272d2945dacc2/Multiscale-climate-emulator-of-multimodal-wave-spectra-MUSCLE-spectra.pdf" target="_blank">(Rueda et al. 2017)</a>.**
# <br>
# 
# **Each cluster is defined by a prototype and formed by the data for which the prototype is the nearest.**
# 
# **Finally the best match unit (bmus) of daily clusters are reordered into a lattice following a geometric criteria, so that similar clusters are placed next to each other for a more intuitive visualization.**

# In[5]:


# PCA data
variance = xds_PCA.variance.values[:]
EOFs = xds_PCA.EOFs.values[:]
PCs = xds_PCA.PCs.values[:]

var_anom_std = xds_PCA.pred_std.values[:]
var_anom_mean = xds_PCA.pred_mean.values[:]
time = xds_PCA.time.values[:]

variance = xds_PCA.variance.values
percent = variance / np.sum(variance)*100
percent_ac = np.cumsum(percent)
n_comp_95 = np.where(percent_ac >= 95)[0][0]
n_comp_90 = np.where(percent_ac >= 90)[0][0]
    
# plot
n_comp = n_comp_90

nterm = n_comp_90 + 1 #n_comp_90 es el número de PC que explican el 90% de la varianza, que en este caso son 237
PCsub = PCs[:, :nterm]
EOFsub = EOFs[:nterm, :]

# normalization
data = PCsub
data_std = np.std(data, axis=0)
data_mean = np.mean(data, axis=0)

#normalize but keep PCs weigth
data_norm = np.ones(data.shape)*np.nan
for i in range(PCsub.shape[1]):
    data_norm[:,i] = np.divide(data[:,i] - data_mean[i], data_std[0]) #si no usas la desviación estándar del primero da mucho error

#KMEANS
num_clusters = 49
kma = KMeans(n_clusters=num_clusters, n_init=1000).fit(data_norm)
kma

#store
with open(r'/home/administrador/Documentos/seasonal/seasonal_forecast/new/kma_model/kma_index_okb.pkl', "wb") as f:
    pickle.dump(kma, f)
    
#a measure of the error of the algorithm
kma.inertia_
# In[6]:


xds_kma_ord,xds_kma = func_kma_order(path_p,xds_PCA,xs)


# In[66]:


xds_kma_sel = trmm_kma(xds_kma,xs_trmm)


# <br>
# 
# **The resulting classification can be seen in the PCs space of the predictor index data. The obtained centroids (black dots), span the wide variability of the data.**

# In[7]:


# CHECKUP PLOT 3D
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plot_3D_kmeans(xds_kma_ord, xds_kma_ord.bmus.values, xds_kma_ord.cenEOFs.values, size_l=12, size_h=10);


# In[8]:


fig = plot_grid_kmeans(xds_kma_ord, xds_kma_ord.bmus.values, xds_kma_ord.cenEOFs.values, 9, ibmus=range(49), size_l=20, size_h=18);


# <br />
# 
# ### <font color='cornflowerblue'>**DWTs plotting**</font> <a name="plot"></a>

# In[6]:


#load
xds_kma = xr.open_dataset(path_p+'kma_model/xds_kma_index_vars_1b.nc')


# In[7]:


path_st = r'/home/administrador/Documentos/'
xds_ibtracs, xds_SP = storms_sp(path_st)


# In[9]:


st_bmus,st_lons,st_lats, st_categ = st_bmus(xds_SP,xds_kma)


# In[9]:


# custom colorbar for index
color_ls = ['white','cyan','cornflowerblue','darkseagreen','olivedrab','gold','darkorange','orangered','red','deeppink','violet','darkorchid','purple','midnightblue']
custom_cmap = custom_colorp(color_ls)


# <br>
# 
# **DWTs lattice and corresponding colors:**

# In[11]:


fig_dwt_lattice = Plot_dwts_colormap(xds_kma.n_clusters.size)


# <br>
# 
# **The resulting clustering of the index predictor, each cell is the mean of all the patterns of the corresponding cluster:**

# In[12]:


fig = Plot_DWTs_Mean_Anom(xds_kma, xs, ['index'], minis=[0], maxis=[.85],levels=[len(color_ls)], kind='mean',cmap = [custom_cmap],genesis='on', st_bmus=st_bmus, 
                          st_lons=st_lons, st_lats=st_lats, markercol='white', markeredge='k');


# <br>
# 
# **DWTs (index) total mean:**

# In[13]:


fig = Plot_DWTs_totalmean(xs, ['index'], minis=[0], maxis=[.85],levels=[len(color_ls)],cmap=[custom_cmap]);


# <br>
# 
# **DWTs (index) Anomalies:**

# In[14]:


fig = Plot_DWTs_Mean_Anom(xds_kma, xs, ['index'], minis=[-.5], maxis=[.5], levels=[20], kind='anom', cmap=['RdBu_r'], 
                          genesis='on', st_bmus=st_bmus, st_lons=st_lons, st_lats=st_lats, markercol='mediumspringgreen', markeredge='k');


# <br />
# <br />
# 
# ### <font color='cornflowerblue'>**Cluster homegeinity**</font> <a name="ch"></a>

# **The DWTs are different to each other, showing the high variability of the data space. The clusters are also very homogenous inside. This confirms 49 as a good choice, which additionally is a manageable number.**

# In[10]:


fig_41 = plot_41(xs,xds_kma)


# In[29]:


fig_5 = plot_5(xs,xds_kma)


# <br />
# <br />
# 
# ### <font color='cornflowerblue'>**DWTs plotting with predictand variables**</font> <a name="plotp"></a>

# In[23]:


fig = Plot_DWTs_Mean_Anom(xds_kma, xs, ['sst', 'dbss'], minis=[22, 0], maxis=[32, 200], levels=[(32-22)/0.5, 8], kind='mean', 
                          cmap=[colorp(), 'seismic'], genesis='on', st_bmus=st_bmus, st_lons=st_lons, st_lats=st_lats, markercol='white', markeredge='k');


# <br>
# 
# **DWTs - SST and MLD total mean:**

# In[24]:


fig = Plot_DWTs_totalmean(xs, ['sst','dbss'], minis=[22,0], maxis=[32,200], levels=[(32-22)/0.5,20],  cmap=[colorp(),'seismic']);


# <br>
# <br>
# 
# **DWTs - SST and MLD Anomalies:**

# In[11]:


fig = Plot_DWTs_Mean_Anom(xds_kma, xs, ['sst', 'dbss'], minis=[-4, -60], maxis=[4, 60], levels=[20, 6], kind='anom', cmap=['coolwarm', 'seismic'], genesis='on', 
                          st_bmus=st_bmus, st_lons=st_lons, st_lats=st_lats, markercol='mediumspringgreen', markeredge='k');


# <br>
# <br>
# 
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: rgb(0,0,0); background-color: #fcf8e3; border-color: #faebcc; ">
#     
# 
# **Clear patterns can be extracted from these figures related to TCs genesis. Most of it takes place under the following conditions:**
# * SST interval from 28ºC to 30ºC (specially 28.5 to 29.5 ºC) that correspond to positive or zero SST anomalies.<br>
# * MLD values equal or smaller to 75 m that correspond to negative anomalies.
#  
# </div>

# <br />
# <br />
# 
# ## <font color='royalblue'>**DWTs seasonality, annual variability and chronology**</font> <a name="chrono"></a>

# **Several plots are shown to better analyse the distribution of DWTs, their transition, persistence and conditioning to TCs occurrence and to AWT.**

# In[48]:


path_mjo_awt = r'/home/administrador/Documentos/STORMS_VIEWER/'


# In[49]:


awt,mjo,awt0 = awt_mjo_ds(path_mjo_awt)


# <br>
# 
# **Seasonality:**

# In[60]:


bmus_DWT, bmus_time,awt0_sel_bmus,bmus_AWT,bmus_MJO = bmus_dwt_mjo(mjo,awt,awt0,xds_kma)


# **TCs genesis according to month:**
# <br>
# 
# ![TITLE](st0.png)

# In[15]:


fig = Plot_DWTs_Probs(bmus_DWT, bmus_time, 49, height=10, width=18);


# In[13]:


#all categories
xs_dwt_counts = dwt_tcs_count_tracks(xds_kma, df, dx=8, dy=8, lo0=lo_area[0], lo1=lo_area[1], la0=la_area[0], la1=la_area[1])
xs_dwt_counts.to_netcdf(r'/home/administrador/Documentos/seasonal/seasonal_forecast/new/kma_model/xds_count_tcs8.nc')

#category 3 
xs_dwt_counts_964 = dwt_tcs_count_tracks(xds_kma, df, dx=8, dy=8, categ=965,lo0=lo_area[0], lo1=lo_area[1], la0=la_area[0], la1=la_area[1])
xs_dwt_counts_964.to_netcdf(r'/home/administrador/Documentos/seasonal/seasonal_forecast/new/kma_model/xds_count_tcs8_964.nc')

# category 2
xs_dwt_counts_979 = dwt_tcs_count_tracks(xds_kma, df, dx=8, dy=8, categ=979,lo0=lo_area[0], lo1=lo_area[1], la0=la_area[0], la1=la_area[1])
xs_dwt_counts_979.to_netcdf(r'/home/administrador/Documentos/seasonal/seasonal_forecast/new/kma_model/xds_count_tcs8_979.nc')


# In[7]:


xs_dwt_counts = xr.open_dataset(path_p+'kma_model/xds_count_tcs8.nc')
xs_dwt_counts_964 = xr.open_dataset(path_p+'kma_model/xds_count_tcs8_964.nc')
xs_dwt_counts_979 = xr.open_dataset(path_p+'kma_model/xds_count_tcs8_979.nc')


# In[8]:


xds_timeline = ds_timeline(df,xs_dwt_counts,xs_dwt_counts_964,xds_kma)


# In[10]:


mask_bmus_YD, mask_tcs_YD = variables_dwt_super_plot(xds_kma, xds_timeline)


# <br>
# 
# **AWT transferred to the DWTs:**

# In[62]:


n_clusters_AWT = 6
n_clusters_DWT = 49
n_clusters_MJO = 8
fig = Plot_Probs_WT_WT(bmus_AWT, bmus_DWT, n_clusters_AWT, n_clusters_DWT, ttl = 'DWT/AWT',height = 15, width = 3, wt_colors=True)


# **Chronology**:

# In[63]:


fig = Report_Sim(xds_kma, py_month_ini=1);


# <br> 
# 
# **Chronology during all the calibration period, with the AWT on the left and the TC days included as black dots:**

# In[22]:


fig = Chrono_dwts_tcs(xds_kma, mask_bmus_YD, mask_tcs_YD, awt0_sel_bmus);


# <br>
# <br>
# 
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: rgb(0,0,0); background-color: #fcf8e3; border-color: #faebcc; ">
#     
# 
#     
# **During the TC season months (November, December, January, February, March and April) the DWTs proability is focused on the upper half of the upper part of the lattice, where most of the TC genesis is also concentrated. In the most intense months (January, February and March) DWTs with the highest number of TCs genesis points are especially likely. On the contrary, in the rest of the months, the probability is shared amongst the DWTs of the lower half of the lattice, where there is very few or null TC genesis activity.**
# 
# </div>

# <br>
# <br>
# 
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: rgb(0,0,0); background-color: #fcf8e3; border-color: #faebcc; ">
#     
# 
# **Intra annual variability:**
# * Months out of the TCs season: purple, pink, gray and blue -> DWTs from 29 to 49 -> low or null TCs genesis activity.<br>
# * Months out of the TCs season: green, orange, red, yellow -> DWTs from 1 to 28 -> high TCs genesis acitvity.
#  
# </div>

# <br>
# <br>
# 
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: rgb(0,0,0); background-color: #fcf8e3; border-color: #faebcc; ">
#     
# 
# **Interanual varibility:**
# * **El Niño**: not much variability, high TC activity (DWTs like 6 or 18 are very probable), 1997 is the year when the season starts earlier (October) and ends later (June)
# * Other factors influences such as long-terms trends, maybe associated to SST warming during this time period.  
#  
# </div>

# <br />
# <br />
# <br />
# 
# ## <font color='royalblue'>**Relationship predictor-predictand**</font> <a name="pp"></a>
# <br>

# ### <font color='cornflowerblue'>**DWTs storm frequency and track counting**</font> <a name="count"></a>
# 
# <br />

# **DWTs - TCs tracks according to genesis**

# In[17]:


fig = Plot_DWTs_tracks(xds_kma, xs, st_bmus, st_lons, st_lats, st_categ, mode='genesis', cocean='lightgray');


# <br>
# 
# **DWTs - TCs tracks segments**

# In[42]:


fig = Plot_DWTs_tracks(xds_kma, xs, st_bmus, st_lons, st_lats, st_categ, mode='segments', cocean='lightgray');


# <br>
# <br>
# 
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: rgb(0,0,0); background-color: #fcf8e3; border-color: #faebcc; ">
#     
# 
# **Considering only the genesis point as the transfer criteria to the DWTs make us miss some important information since the TCs tracks develop along many days, during which the DWT can change.**
# 
# </div>

# **The predictor area is discretized in <u>squared 8º cells</u> to compute the storm frequency per DWT.**

# **Absolute number of TCs per DWT:**

# In[15]:


fig_8 = Plot_DWTs_counts(xs_dwt_counts, mode='counts_tcs');


# <br>
# 
# **Absolute number of TCs reaching category 2 or greater per DWT:**

# In[16]:


fig_8 = Plot_DWTs_counts(xs_dwt_counts_979, mode='counts_tcs');


# <br>
# 
# **Number of TCs per day conditioned to each DWT, with DWT probability as background color:**

# In[17]:


fig_8 = Plot_DWTs_counts(xs_dwt_counts, mode='tcs_dwts');


# <br>
# 
# **Number of TCs reaching category 2 or greater per day conditioned to each DWT, with DWT probability as background color:**

# In[18]:


fig_8 = Plot_DWTs_counts(xs_dwt_counts_979, mode='tcs_dwts');


# <br />
# <br />
# 
# ### <font color='cornflowerblue'>**Calibration time period predictand plotting**</font> <a name="cali"></a>
# 
# <br>

# 
# 
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: rgb(0,0,0); background-color: #fcf8e3; border-color: #faebcc; ">
#     
# 
# **Recall:**
# * Generation of the index predictor.
# * Clustering of the index predictor in 49 synoptic patterns named DWTs.
# * Calculation of the number of TCs per day conditioned to each DWT in 8x8º cells in the target area. 
# 
# **Each day of the calibration period has therefore its <u>expected number of TCs map.</u>**
#     
# </div>

# In[19]:


xds_timeline = ds_timeline(df,xs_dwt_counts,xs_dwt_counts_964,xds_kma)


# <br>
# 
# **This mean expected daily number of TCs in 8x8º cells is aggregated in monthly basis for an easy management and visualization.**

# In[9]:


# resample months
xds_timeM0 = xds_timeline.resample(time='MS', skipna=True).sum()
del xds_timeM0['bmus']
xds_timeM = xds_timeM0.where(xds_timeM0.probs_tcs > 0, np.nan)
#xds_timeM.to_netcdf(path_p+'xds_timeM8.nc')
xds_timeM


# <br>
# 
# **Expected mean number of TCs in 8x8º cells in the target area for the calibration time period (1982-2019), including the historical TC tracks:**

# In[23]:


fig_cali_8 = plot_calibration_period(xds_timeM,xds_timeline,df,1)


# <br>
# 
# **Expected number of TCs reaching category 3 or greater in 8x8º cells in the target area for the calibration time period (1982-2019), including the historical TC tracks:**

# In[12]:


fig_cali_8 = plot_calibration_period_cat3(xds_timeM,xds_timeline,df,0.6,8)


# <br>
# 
# **Expected number of TCs in 8x8º cells in the target area for year 2015 (El Niño) of the calibration time period (1982-2019) including the historical TC tracks coloured according to their category:**

# In[10]:


fig_year_8 = plot_cali_year(2015,xds_timeline,xds_timeM,df,35)


# <br>
# <br>
# 
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: rgb(0,0,0); background-color: #fcf8e3; border-color: #faebcc; ">
#     
# 
# 
# * **The model generally performs very well since it effectively and accuretly estimates the TC activity (quantity and intensity of TCs).**
# * **In some zones (depending on the month) the model overstimates the number of TCs since rather than predicting the exact path followed by the TC track it predicts the track and its area of infuence (surroundings).**
# * **When a TC is very intense or very close in dates to the previous or following month it leaves its footprint (like Pam from 07/03/2015 in February).**
# 
# </div>
# 
