#/bin/sh +ex

mkdir -p results/nextstrain

curl https://data.nextstrain.org/files/ncov/open/metadata.tsv.gz -o results/nextstrain/metadata.tsv.gz

gunzip -kvf results/nextstrain/metadata.tsv.gz
