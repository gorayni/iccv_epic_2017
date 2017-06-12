#!/bin/bash 
tput reset

if [ -z "$1" ]
then
  TIMESTEP=10  
else
  TIMESTEP=$1
fi

nohup python -u VGG-16+LSTM_training.py --timestep $TIMESTEP </dev/null 2>&1 | tee "VGG-16+LSTM_training.timestep_"$TIMESTEP"."`date +%Y-%m-%d_%H:%M:%S`".log" &
