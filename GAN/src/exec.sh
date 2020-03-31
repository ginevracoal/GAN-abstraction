#!/bin/bash

MODEL="eSIR"  # SIR, eSIR, Repress, Toggle
TIMESTEPS=32
NOISE_TIMESTEPS=128
BATCH_SIZE=128
EPOCHS=100
GEN_EPOCHS=20
GEN_LR=0.0001
DISCR_LR=0.0001
FIXED_PARAMS=1
N_TRAJ=30000

### launch from anywhere on server
cd ~/GAN-abstraction/GAN/src/

## activate venv
source ~/GAN-abstraction/GAN/venv_gan/bin/activate

## set filenames
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
RESULTS="../results/$DATE/"	
mkdir -p $RESULTS
OUT="${RESULTS}${TIME}_out.txt"

## run script
python3 gan_abstraction.py --traj=$N_TRAJ --gen_lr=$GEN_LR --discr_lr=$DISCR_LR --timesteps=$TIMESTEPS --noise_timesteps=$NOISE_TIMESTEPS --model=$MODEL --epochs=$EPOCHS --gen_epochs=$GEN_EPOCHS --fixed_params=$FIXED_PARAMS --batch_size=$BATCH_SIZE &>> $OUT
python3 gan_evaluation.py --traj=20 --gen_lr=$GEN_LR --discr_lr=$DISCR_LR --timesteps=$TIMESTEPS --noise_timesteps=$NOISE_TIMESTEPS --model=$MODEL --epochs=$EPOCHS --gen_epochs=$GEN_EPOCHS --fixed_params=$FIXED_PARAMS &>> $OUT

## deactivate venv
deactivate