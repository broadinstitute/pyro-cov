#!/usr/bin/env python
# coding: utf-8

# Preprocess files for parsing

# In[1]:


import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="generate_epiToPublicAndDate")
parser.add_argument("--metadata-file-basename")
args = parser.parse_args()



# In[2]:

filename = "results/gisaid/" + args.metadata_file_basename + ".tsv.gz"
gisaid_meta = pd.read_csv(filename, sep="\t")


# In[3]:


gisaid_meta["vname"] = gisaid_meta["Virus name"].str.replace("hCoV-19/","")
gisaid_meta["vname2"] = gisaid_meta["vname"]


# In[4]:


epi_map = gisaid_meta[["Accession ID", "vname", "vname2", "Collection date"]]


# In[5]:


epi_map = epi_map.sort_values(by="Accession ID", ascending = True)


# In[6]:


epi_map


# In[7]:


epi_map.to_csv("results/gisaid/epiToPublicAndDate.latest", header=False, sep="\t", index=False)


# In[8]:


# get_ipython().run_line_magic('pinfo', 'pd.to_csv')


# In[9]:


# get_ipython().run_line_magic('pinfo', 'pd.DataFrame.to_csv')


# In[ ]:
