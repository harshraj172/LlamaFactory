#!/bin/bash
#SBATCH --job-name=llama_fsdp_sft
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --account=transfernetx
#SBATCH --partition=booster
#SBATCH --time=24:00:00
#SBATCH --threads-per-core=1
#SBATCH --mem=400GB
#SBATCH --output=/p/project1/laionize/harsh/LLaMA-Factory/slurm_output/sft_fsdp_%j.out

set -euo pipefail

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

export CONTAINER=/p/scratch/laionize/raj3/containers/llamafactory_leonardo.sif

# HF VARS
export HF_HOME=/p/scratch/laionize/raj3/.cache
export TRANSFORMERS_CACHE=/p/scratch/laionize/raj3/.cache
export HF_DATASETS_CACHE=/p/scratch/laionize/raj3/.cache
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# DISTRIBUTED SETUP
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # master node hostname
# MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"  # IP numeric address
export MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 2000))
echo "MASTER_ADDR:MASTER_PORT set to: ${MASTER_ADDR}:${MASTER_PORT}"

export GPUS_PER_NODE=4
export WORLD_SIZE=$(( SLURM_NNODES * 4 ))
export RANK=${SLURM_NODEID}

# NCCL settings to improve distributed training stability (handling flipping links, irresponsive nodes, etc)
# waiting for 120s in case nodes become irresponsive giving a chance to recover
export NCCL_IB_TIMEOUT=120
export NCCL_DEBUG=INFO
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

echo "Starting training..."
echo "=========================================="

# srun --wait=60 --kill-on-bad-exit=1 \
#   singularity exec --nv \
#   --bind /leonardo_work/AIFAC_L01_028:/leonardo_work/AIFAC_L01_028 \
#   "$CONTAINER" \
#   accelerate launch --config_file examples/accelerate/fsdp_config_offload.yaml \
#   llamafactory-cli train examples/train_full/opensci_full_sft_fsdp.yaml

export SINGULARITYENV_MASTER_ADDR="$MASTER_ADDR"
export SINGULARITYENV_MASTER_PORT="$MASTER_PORT"
export SINGULARITYENV_GPUS_PER_NODE="$GPUS_PER_NODE"
export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export SINGULARITYENV_HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE"
export SINGULARITYENV_TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
export SINGULARITYENV_HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
export SINGULARITYENV_NCCL_IB_TIMEOUT="$NCCL_IB_TIMEOUT"
export SINGULARITYENV_NCCL_DEBUG="$NCCL_DEBUG"
export SINGULARITYENV_TRITON_LIBCUDA_PATH="$TRITON_LIBCUDA_PATH"
export SINGULARITYENV_CUDA_DEVICE_MAX_CONNECTIONS="$CUDA_DEVICE_MAX_CONNECTIONS"
export SINGULARITYENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"

srun --export=ALL --wait=60 --kill-on-bad-exit=1 \
  singularity exec --nv \
  --bind /p/project1/laionize/harsh:/p/project1/laionize/harsh \
  "$CONTAINER" \
  bash -lc '
    set -e
    
    export PATH="/opt/conda/bin:$PATH"
    export NODE_RANK="$SLURM_NODEID"
    export NNODES="$SLURM_NNODES"
    export HF_HOME=/p/scratch/laionize/raj3/.cache
    export ALLOW_EXTRA_ARGS=1 # for rope_theta

    cd /p/project1/laionize/harsh/LLaMA-Factory

    llamafactory-cli train examples/train_full/opensci_full_sft_fsdp.yaml
  '