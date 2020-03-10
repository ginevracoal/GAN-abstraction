#!/bin/bash

MODEL="eSIR"  # SIR, eSIR, Repress, Toggle
TIMESTEPS=128
NOISE_TIMESTEPS=128
BATCH_SIZE=512
EPOCHS=50
GEN_EPOCHS=20
EMBED=False
FIXED_PARAMS=True

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
python3 gan_abstraction.py --n_traj=30000 --timesteps=$TIMESTEPS --noise_timesteps=$NOISE_TIMESTEPS --batch_size=$BATCH_SIZE --model=$MODEL --epochs=$EPOCHS --gen_epochs=$GEN_EPOCHS --embed=$EMBED --fixed_params=$FIXED_PARAMS&> $OUT
python3 gan_evaluation.py --n_traj=2000 --timesteps=$TIMESTEPS --noise_timesteps=$NOISE_TIMESTEPS --model=$MODEL --epochs=$EPOCHS --gen_epochs=$GEN_EPOCHS &> $OUT

## deactivate venv
deactivate