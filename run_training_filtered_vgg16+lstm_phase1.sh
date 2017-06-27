#!/bin/bash 
tput reset

if [ -z "$1" ]
then
  TIMESTEP=10  
else
  TIMESTEP=$1
fi

if [ -z "$2" ]
then
  OVERLAP=2
else
  OVERLAP=$2
fi

nohup python -u Filtered_VGG-16+LSTM_phase1_training.py --timestep $TIMESTEP --overlap $OVERLAP </dev/null 2>&1 | tee "Filtered_VGG-16+LSTM_training.timestep_"$TIMESTEP".overlap_"$OVERLAP"."`date +%Y-%m-%d_%H:%M:%S`".log" &
