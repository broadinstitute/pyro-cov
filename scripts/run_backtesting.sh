#!/bin/bash

backtesting_days=$(seq -s, 150 14 550)

python mutrans.py --backtesting-max-day $backtesting_days --forecast-steps 12
