#!/bin/bash

## Ben Kotzen
## bkotzen@mgh.hardvard.edu

# Assume that GISAID metadata files have already been downloaded


if [[ "#$" -ne 2 ]]; then
echo ""
echo "--------------------------------------------------------"
echo " Arg 1 - date "
echo " Arg 2 - Angie's tree url "

# Create and link directories
# cd pyro-cov
# d=$(date +%Y-%m-%d)
d = $1
# mkdir results.”$d”
rm results
ln -sf results.$d results

# Download Angie's tree
# mkdir results/gisaid && cd results/gisaid
cd results/gisaid
usher_url = $2
usher_file=$(basename "$usher_url")
wget “$usher_url”
cd ../..

# Download metadata file
# cd results/gisaid
# fileid = $3
# gdown $fileid
# cd ../..

mfb = "metadata_$d"
ipython generate_epiToPublicAndDate.py --metadata-file-basename 
