#!/bin/sh -ex

# Ensure data directory (or a link) exists.
test -e data || mkdir data

# Download and decompress data.
curl -u $GISAID_USERNAME:$GISAID_PASSWORD \
  https://www.epicov.org/epi3/3p/$GIAID_FEED/export/provision.json.xz \
  | xz -d -T8 > data/gisaid.json
