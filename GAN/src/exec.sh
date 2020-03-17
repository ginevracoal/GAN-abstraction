#!/bin/bash

MODEL="eSIR"  # SIR, eSIR, Repress, Toggle
TIMESTEPS=128
NOISE_TIMESTEPS=128
BATCH_SIZE=128
EPOCHS=40
GEN_EPOCHS=40
EMBED=1
FIXED_PARAMS=0
LR=0.001 

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
if [ $FIXED_PARAMS -eq 0 ]; then N_TRAJ=30000; elif [ $FIXED_PARAMS -eq 1 ]; then N_TRAJ=2500; fi

python3 gan_abstraction.py --n_traj=$N_TRAJ --lr=$LR --timesteps=$TIMESTEPS --noise_timesteps=$NOISE_TIMESTEPS --batch_size=$BATCH_SIZE --model=$MODEL --epochs=$EPOCHS --gen_epochs=$GEN_EPOCHS --embed=$EMBED --fixed_params=$FIXED_PARAMS &> $OUT
python3 gan_evaluation.py --n_traj=$N_TRAJ --lr=$LR --timesteps=$TIMESTEPS --noise_timesteps=$NOISE_TIMESTEPS --model=$MODEL --epochs=$EPOCHS --gen_epochs=$GEN_EPOCHS --embed=$EMBED --fixed_params=$FIXED_PARAMS &> $OUT

## deactivate venv
deactivate