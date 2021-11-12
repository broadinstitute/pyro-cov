#!/bin/sh -ex

# Ensure data directory (or a link) exists.
test -e results || mkdir results

# Download.
curl -u $GISAID_USERNAME:$GISAID_PASSWORD --retry 4 \
  https://www.epicov.org/epi3/3p/$GISAID_FEED/export/provision.json.xz \
  > results/gisaid.json.xz

# Decompress, keeping the original.
xz -d -k -f -T0 -v results/gisaid.json.xz
