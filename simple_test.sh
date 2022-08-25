#!/bin/bash

# Gather arguments
d=$1
url=$2
subset=$3

# Test all args
usher_url=$url
usher_file=$(basename "$usher_url")
wget -O "metadata_subsets/$subset/results.$d/gisaid/$usher_file" "$usher_url"
