#!/bin/bash

## Ben Kotzen
## bkotzen@mgh.hardvard.edu

# Assume that GISAID metadata files have already been downloaded


if [[ "#$" -ne 2 ]]; then
echo ""
echo "--------------------------------------------------------"
echo " Arg 1 - date (YYYY-mm_dd)"
echo " Arg 2 - Angie's tree url "
echo " Arg 3 - metadata subset  "
echo "--------------------------------------------------------"

d=$1
url=$2
subset=$3

# Create and link directories
echo "Creating and linking directories"
# cd pyro-cov
# d=$(date +%Y-%m-%d)
# mkdir results.”$d”
rm results
ln -sf metadata_subsets/$subset/results.$d results

# Download Angie's tree
echo "Downloading UShER tree"
# mkdir results/gisaid && cd results/gisaid
usher_url=$url
usher_file=$(basename "$usher_url")
wget -O "results/gisaid/$usher_file" "$usher_url"

# Download metadata file
# cd results/gisaid
# fileid = $3
# gdown $fileid
# cd ../..

# Process phylogenies
echo "Processing phylogenies"
mfb="metadata_$d"
ipython generate_epiToPublicAndDate.py --metadata-file-basename mfb
python scripts/preprocess_usher.py --tree-file-in results/gisaid/"$usher_file" --gisaid-metadata-file-in results/gisaid/"$mfb".tsv.gz

# Fit model
echo "Fitting model"
nohup python scripts/mutrans.py > results/nohup.out 2>&1 &
