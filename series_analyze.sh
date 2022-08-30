#!/bin/bash

## Ben Kotzen
## bkotzen@mgh.hardvard.edu

echo ""
echo "--------------------------------------------------------"
echo "This script runs the PyR0 model given the data that was "
echo "historically available at specific points in time:      "
echo "    -The first observed date of each VoC:               "
echo "       (Alpha, Beta, Delta, BA.1, BA.2, BA.2.12.1, BA.4)"
echo "    -One week, two weeks, and four weeks after observing"
echo "      each of these variants                            "
echo "In order to run the PyR0 model, we utilize analyze.sh   "
echo ""
echo "--------------------------------------------------------"

declare -a series=(\
# "2022-02-10 BA.2.12.1_wk0"\
"2022-02-17 https://hgwdev.gi.ucsc.edu/~angie/9620db8/gisaidAndPublic.2022-02-17.masked.pb.gz BA.2.12.2_wk1 "\
"2022-02-24 https://hgwdev.gi.ucsc.edu/~angie/6bca5eb/gisaidAndPublic.2022-02-23.masked.pb.gz BA.2.12.1_wk2 "\
"2022-03-10 https://hgwdev.gi.ucsc.edu/~angie/14b4e4e/gisaidAndPublic.2022-03-10.masked.pb.gz BA.2.12.1_wk4 "\
)

for args in ${series[@]}; do
  echo $args
done
