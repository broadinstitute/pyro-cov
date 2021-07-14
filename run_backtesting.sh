#!/bin/bash

backtesting_days=$(seq -s, 150 7 550)
echo $backtesting_days
#python mutrans.py --backtesting-max-day 150,200,250,300,350,400,450,500,550
python mutrans.py --backtesting-max-day $backtesting_days
