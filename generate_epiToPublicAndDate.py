# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python
# coding: utf-8

# Preprocess files for parsing

# In[1]:


import pandas as pd

# In[2]:


gisaid_meta = pd.read_csv("results/gisaid/metadata_2022_08_08.tsv.gz", sep="\t")


# In[3]:


gisaid_meta["vname"] = gisaid_meta["Virus name"].str.replace("hCoV-19/", "")
gisaid_meta["vname2"] = gisaid_meta["vname"]


# In[4]:


epi_map = gisaid_meta[["Accession ID", "vname", "vname2", "Collection date"]]


# In[5]:


epi_map = epi_map.sort_values(by="Accession ID", ascending=True)


# In[6]:


epi_map


# In[7]:


epi_map.to_csv(
    "results/gisaid/epiToPublicAndDate.latest", header=False, sep="\t", index=False
)


# In[8]:


# get_ipython().run_line_magic('pinfo', 'pd.to_csv')



# In[ ]:
