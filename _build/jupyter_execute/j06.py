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
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image

#lib
from lib.validation_methodology_plots import *


# In[2]:


path_p = r'/home/administrador/Documentos/seasonal/seasonal_forecast/new/'


# In[3]:


df_2021 = pd.read_pickle(path_p+'df_coordinates_pmin_sst_mld_2021.pkl')
xs = xr.open_dataset(path_p+'xs_index_vars_19822019_2deg_new.nc')
xds_kma = xr.open_dataset(path_p+'kma_model/xds_kma_index_vars_1b.nc')
xs_dwt_counts = xr.open_dataset(path_p+'kma_model/xds_count_tcs8.nc')
xs_dwt_counts_964 = xr.open_dataset(path_p+'kma_model/xds_count_tcs8_964.nc')
xds_timeM = xr.open_dataset(path_p+'xds_timeM8.nc')
xds_PCA = xr.open_dataset(path_p+'xds_PCA.nc')
xds_kma_ord = xr.open_dataset(path_p+'xds_kma_ord.nc')


# <br>
# <br>
# <br>
# 
# #  <font color='navy'>**Model Validation** </font> 
# 

# >[Index predictor](#p)<br> <br>
# >[Cluster comparison](#cc)<br> <br>
# >[Predictand computation and plotting](#plv)<br> <br>
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

# <br>
# <br>
# 
# **After analizing the tailor-made predictor along the hindcast data for the calibration period (1982-2019), the performace of the model will be validated for year 2020, which has not been included in the predictor calibration process.**

# <br>
# 
# 
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: rgb(0,0,0); background-color: #fcf8e3; border-color: #faebcc; ">
#     
# 
# **Steps:**
# * **1.** Download and preprocess (file conversion and resolution interpolation) SST and MLD data for the validation time period.
# * **2.** Generation of the index predictor based on the index function obtained at the calibration period.
# * **3.** The fitted Principal Component Analysis for the calibration is used to predict the index principal components in that same temporal-spatial space.
# * **4.** The predicted PCs are assigned to the best match unit group from the fitted K-means clustering -> based on the index predictor a DWT is assigned to each day.
# * **5.** From the DWT the expected daily mean number of TCs in 8x8º cells map in the target area is known.
#     
# 
# </div>

# <br />
# <br />
# 
# ## <font color='royalblue'>**Index predictor and DWTs**</font> <a name="p"></a>

# 
# 
# **Download and preprocess (file conversion and resolution interpolation) SST and MLD data for the validation time period.**

# In[5]:


path_val = r'/home/administrador/Documentos/seasonal/seasonal_forecast/validation/'
year_val = 2020


# In[5]:


change_sst_resolution_val(path_val,year_val)


# <br>
# 
# **Generation of the index predictor based on the index function obtained at the calibration period.**

# In[6]:


xs_val = ds_index_over_time_val(path_val,path_p,year_val)
xs_val


# <br>
# <br>
# 
# **The fitted Principal Component Analysis for the calibration is used to predict the index principal components in that same temporal-spatial space and the predicted PCs are assigned to the best match unit group from the fitted K-means clustering -> based on the index predictor a DWT is assigned to each day.**

# In[7]:


val_bmus = PCA_k_means_val(path_p,path_val,xs_val)


# <br>
# <br>
# 
# **Chronology of the DWTs:**

# In[16]:


fig_bmus = plot_bmus_chronology(xs_val,val_bmus,year_val)


# <br>
# 
# **The resulting classification can be seen in the PCs space of the predictor index data. The obtained centroids (black dots), span the wide variability of the data.**

# In[17]:


fig = plot_scatter_kmeans(xds_kma_ord, val_bmus, xds_kma_ord.cenEOFs.values, size_l=12, size_h=10);


# <br />
# <br />
# 
# ## <font color='royalblue'>**Cluster comparison**</font> <a name="cc"></a>

# In[9]:


fig = plot_bmus_comparison_validation_calibration(xs,xds_kma,xs_val,val_bmus,9,49)


# <br />
# <br />
# 
# ## <font color='royalblue'>**Predictand computation and plotting**</font> <a name="plv"></a>

# **From the DWT the daily expected mean number of TCs in 8x8º cells in the target area is known for each day and thus maps at different time scales can be computed.**

# **Daily mean expected number of TCs**

# In[9]:


xds_timeline_val,xs_M_val = ds_monthly_probabilities_val(df_2021,val_bmus,xs_val,xs_dwt_counts,xs_dwt_counts_964)


# <br>
# 
# **Monthly aggregated mean expected number of TCs**

# In[10]:


xs_M_val


# In[22]:


fig_val_year_8 = plot_validation_year(df_2021,xs_M_val,xds_timeline_val,35)


# <br>
# 
# **Whole period aggregated mean expected number of TCs**

# In[46]:


fig_val_year_8 = plot_validation_full_season(df_2021,xs_M_val,xds_timeline_val,35)


# <br>
# <br>
# 
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: rgb(0,0,0); background-color: #fcf8e3; border-color: #faebcc; ">
#     
# 
# * **The model performs very well when estimating the expected TC activity (number and intensity of TCs), not understimating the threat.**    
#     
# * **In some cells adjacents to the cells including TC tracks it overstimates TC activity.**
#     
# </div>
