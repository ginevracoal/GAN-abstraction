#!/bin/bash

N_TRAJ=100
TIMESTEPS=128
NOISE_TIMESTEPS=3
BATCH_SIZE=25
MODEL="SIR"
EPOCHS=100
LR=0.02


### launch from anywhere on server
cd ~/GAN-abstraction/GAN/src/

## activate venv
source ~/GAN-abstraction/GAN/venv_gan/bin/activate

## set filenames
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
RESULTS="../results/$DATE/"
mkdir -p $RESULTS
OUT="${RESULTS}${TIME}_${DATASET_NAME}_out.txt"

## run script
python3 gan_abstraction.py -n=$N_TRAJ -t=$TIMESTEPS --noise_timesteps=$NOISE_TIMESTEPS --batch_size=$BATCH_SIZE --model=$MODEL --epochs=$EPOCHS --lr=$LR &> $OUT

## deactivate venv
deactivate