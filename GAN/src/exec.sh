#!/bin/bash

N_TRAJ=30000
TIMESTEPS=118
NOISE_TIMESTEPS=5
BATCH_SIZE=32
MODEL="SIR" # SIR, eSIR
EPOCHS=300
GEN_EPOCHS=10

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
python3 gan_abstraction.py -n=$N_TRAJ -t=$TIMESTEPS --noise_timesteps=$NOISE_TIMESTEPS --batch_size=$BATCH_SIZE --model=$MODEL --epochs=$EPOCHS --gen_epochs=$GEN_EPOCHS &> $OUT

## deactivate venv
deactivate