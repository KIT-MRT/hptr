#!/bin/bash
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.out
#SBATCH --time=120:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5000
#SBATCH --tmp=200000
#SBATCH --gpus=rtx_2080_ti:4
#SBATCH --open-mode=truncate

trap "echo sigterm recieved, exiting!" SIGTERM

export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET_DIR="/mrtstorage/datasets_tmp/waymo_hptr"
run () {
    python -u hptr_modules/run.py \
    trainer.limit_train_batches=0.25 \
    trainer.limit_val_batches=0.25 \
    trainer=womd \
    model=scg_womd \
    +trainer.devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) \
    +trainer.accelerator=gpu \
    datamodule=h5_womd \
    datamodule.batch_size=3 \
    datamodule.n_agent=64 \
    loggers.wandb.name="hptr_womd" \
    loggers.wandb.project="hptr_train" \
    loggers.wandb.entity="kit-mrt" \
    datamodule.data_dir=$DATASET_DIR \
    +loggers.wandb.offline=False \
    hydra.run.dir='logs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
}

run
