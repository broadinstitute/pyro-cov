#!/bin/bash

## Ben Kotzen
## bkotzen@mgh.hardvard.edu

if [[ "#$" -ne 2 ]]; then
echo ""
echo "--------------------------------------------------------"
echo " Arg 1 - voc"
echo " Arg 2 - date (YYYY-mm-dd)"
echo "--------------------------------------------------------"
fi

voc=$1
d=$2

# Create and link directories
echo ANALYZE_EPOCH.SH "Creating and linking directories"
# cd pyro-cov
# d=$(date +%Y-%m-%d)
# mkdir results.”$d”
rm results
ln -sf epochs/$voc/results.$d results

# Download Angie's tree
echo ANALYZE_EPOCH.SH "Downloading UShER tree"
# mkdir results/gisaid && cd results/gisaid
usher_url=https://hgwdev.gi.ucsc.edu/~angie/caea02f/gisaidAndPublic.2022-09-14.masked.pb.gz
usher_file=$(basename "$usher_url")
wget -O "results/gisaid/$usher_file" "$usher_url"

# Download metadata file
# cd results/gisaid
# fileid = $3
# gdown $fileid
# cd ../..

# Process phylogenies
echo ANALYZE_EPOCH.SH "Processing phylogenies"
mfb="metadata_$d"
python generate_epiToPublicAndDate.py --metadata-file-basename $mfb
python scripts/preprocess_usher.py --tree-file-in results/gisaid/"$usher_file" --gisaid-metadata-file-in results/gisaid/"$mfb".tsv.gz

# Fit model
echo ANALYZE_EPOCH.SH "Fitting model in background - progress in results/nohup.out"
nohup python scripts/mutrans.py > results/nohup.out 2>&1 &
