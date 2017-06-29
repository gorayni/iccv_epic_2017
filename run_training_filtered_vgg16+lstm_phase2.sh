#!/bin/bash
if [ -z "$1" ]
then
  TIMESTEP=10
else
  TIMESTEP=$1
fi

if [ -z "$2" ]
then
  OVERLAP=3
else
  OVERLAP=$2
fi

tput reset

weights_filepath="weights.Filtered_VGG16+LSTM.phase_1.timesteps_"$TIMESTEP".overlap_"$OVERLAP".best.tf.hdf5"
echo "Filtered VGG-16+LSTM phase 1 training Timestep: "$TIMESTEP" Overlap: "$OVERLAP
echo "Weights filepath: "$weights_filepath
nohup python -u Filtered_VGG-16+LSTM_phase2_training.py --weights "$weights_filepath" --timestep $TIMESTEP --overlap $OVERLAP </dev/null 2>&1 | tee "Filtered_VGG-16+LSTM_training.timestep_"$TIMESTEP".overlap_"$OVERLAP"."`date +%Y-%m-%d_%H:%M:%S`".log" &
