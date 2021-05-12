#!/usr/bin/env python
# coding: utf-8

# In[3]:


#lib
import xarray as xr 
from lib.plots_dwts import colorp,  custom_colorp,Plot_DWTs_Mean_Anom, Plot_DWTs_totalmean,Plot_Probs_WT_WT, Plot_Probs_WT_WT_anomaly, Plot_Probs_WT_WT_WT, Plot_DWTs_Probs, Report_Sim_oneyear, Report_Sim, Plot_DWTs_counts, Chrono_dwts_tcs, Chrono_probs_tcs, Plot_dwts_colormap

from lib.predictor_definition_building import SP_genesis_cat,storms_sp

import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image


# <br>
# <br> 
# <br>
# 
# #  <font color='navy'>**Additional Predictand Variables** </font> 
# 

# > [Sea Level Pressure (SLP)](#slp)<br> <br>
# > [Daily mean precipitation](#cfs)<br> <br><br> 
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
# 

# <br />
# 
# ## <font color='royalblue'>**Sea Level Pressure (SLP)**</font> <a name="slp"></a>
# 
# <br />

# In[8]:


path_p = r'/home/administrador/Documentos/seasonal/seasonal_forecast/new/'

xs = xr.open_dataset(path_p+'xs_index_vars_19822019_2deg_new.nc')
xds_kma = xr.open_dataset(path_p+'kma_model/xds_kma_index_vars_1b.nc')
xs_trmm = xr.open_dataset(path_p+'xs_trmm_1982_2019_2deg_new.nc')
xds_kma_sel = xr.open_dataset(path_p+'kma_model/xds_kma_index_trmm_1b.nc')


# In[10]:


path_st = r'/home/administrador/Documentos/'
xds_ibtracs, xds_SP = storms_sp(path_st)
st_lons = xds_SP.lon.values
st_lats = xds_SP.lat.values


# In[11]:


st_bmus = SP_genesis_cat(xds_SP,xds_kma)


# <br>
# 
# **DWTs - SLP Mean:**
# <br>

# In[5]:


# pressure > 1013mbar (anticiclon)
fig = Plot_DWTs_Mean_Anom(xds_kma, xs, ['slp'], minis=[1003], maxis=[1023], levels=[20], kind='mean', cmap=['RdBu_r'],
                          genesis='on', st_bmus=st_bmus, st_lons=st_lons, st_lats=st_lats, markercol='deeppink', markeredge='k');


# <br>
# <br>
# 
# **DWTs - SLP Anomalies**

# In[7]:


fig = Plot_DWTs_Mean_Anom(xds_kma, xs, ['slp'], minis=[-9], maxis=[9], levels=[20], kind='anom', cmap=['coolwarm'],
                          genesis='on', st_bmus=st_bmus, st_lons=st_lons, st_lats=st_lats, markercol='deeppink', markeredge='k');


# <br />
# <br />
# 
# ## <font color='royalblue'>**Daily Mean Precipitation**</font> <a name="cfs"></a>

# <br>
# 
# 
# **DWTs - Daily Mean Precipiation Mean**

# In[13]:


fig = Plot_DWTs_Mean_Anom(xds_kma_sel, xs_trmm, ['precipitation'],minis=[0], maxis=[30], levels=[20],kind='mean', cmap=['gist_ncar_r'], 
                          genesis='on', st_bmus=st_bmus, st_lons=st_lons, st_lats=st_lats, markercol='white', markeredge='k');


# <br>
# 
# 
# **DWTs - Daily Mean Precipiation Anomalies**

# In[15]:


fig = Plot_DWTs_Mean_Anom(xds_kma_sel, xs_trmm, ['precipitation'], minis=[-20], maxis=[20], levels=[20],kind='anom', cmap=['coolwarm'],
                          genesis='on', st_bmus=st_bmus, st_lons=st_lons, st_lats=st_lats, markercol='mediumspringgreen', markeredge='k');


# <br>
# <br>
# 
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: rgb(0,0,0); background-color: #fcf8e3; border-color: #faebcc; ">
#     
# 
# **Clear patterns can be extracted from these figures related to TCs genesis. Most of it takes place under the following conditions:**
# * Low pressure areas, with 1013 mba or lower values, corresponding generally to negative anomalies.
# * Intense but not extreme precipitation areas, from 9 to 16.5 mm/day, corresponding generally to red anomalies.
#     
# </div>
# 

# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: rgb(0,0,0); background-color: #fcf8e3; border-color: #faebcc; ">
# 
#     
# **These conclusions add to <u>the summary relationship predictand - predictors:**</u> 
# <br>
#     
# * **TC activity is focused in the first 28 DWTs (index clusters).**
# <br>  
# * **The TCs genesis activity is generally focused under the following conditions:**
#     * Index range values from 0.60 to 0.79, corresponding to positive anomalies.
#     * In the warm SST zone, 28 - 30 º C, and where MLD values are smaller than 75 m; corresponding to mild positive SST anomalies and negative MLD anomalies.
#     * In intense but not extreme precipitation areas, from 9 to 16.5 mm/day, corresponding generally to red anomalies.
#     * In low pressure areas, with 1013 mba or lower values, corresponding generally to negative anomalies.
#     
# </div>
