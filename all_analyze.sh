#!/bin/bash

## Ben Kotzen
## bkotzen@mgh.hardvard.edu

voc=$1

declare -a series=("2020-11-18 2020-12-02 2020-12-16 ")
##"2020-11-18 2020-12-02 2020-12-16 " \ ## alpha
##"2021-03-12 2021-03-26 2021-04-09 " \ ## beta
##"2021-03-17 2021-03-31 2021-04-14 " \ ## delta
##"2021-11-08 2021-11-22 2021-12-06 " \ ## BA.1
##"2021-11-13 2021-11-27 2021-12-11 " \ ## BA.2
##"2022-01-27 2022-02-10 2022-02-24 " \ ##B A.2.12.1
##"2022-03-29 2022-04-12 2022-04-26 " \ ## BA.4


for row in "${series[@]}"; do
  args=($row)
  d1=${args[0]}
  d2=${args[1]}
  d3=${args[2]}
      
  echo ALL_ANALYZE.SH analyze_epoch.sh $voc $d1
##  source analyze_epoch.sh $voc $d1
  wait
  echo ALL_ANALYZE.SH ----------------------------------------------------------
  echo ALL_ANALYZE.SH analyze_epoch.sh $voc $d2
##  source analyze_epoch.sh $voc $d2
  wait
  echo ALL_ANALYZE.SH ----------------------------------------------------------
  echo ALL_ANALYZE.SH analyze_epoch.sh $voc $d3
##  source analyze_epoch.sh $voc $d3
  wait
  echo ALL_ANALYZE.SH ----------------------------------------------------------
done
