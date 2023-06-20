#!/bin/bash
#BSUB -J "ell3_corr2_int2_rmax_5_128_equi_II"
#BSUB -o "/usr/workspace/vita1/logs/lsf/%J.out"
#BSUB -e "/usr/workspace/vita1/logs/lsf/%J.err"
#BSUB -G c02red
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 4:00

# Environment setup
module load gcc/8.3.1
module load cuda/11.6.1

# conda init bash
# conda activate conda-development
# source /g/g20/vita1/venv-development/bin/activate
source /usr/workspace/vita1/programs/anaconda/bin/activate
conda activate opence-1.7.2-cuda-11.4

# Used for PyTorch Lightning
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR=$(hostname)

jsrun -n 1 python3 -m ip_explorer.information_imbalance \
    --seed 1123 \
    --num-nodes 1 \
    --gpus-per-node 1 \
    --batch-size 2 \
    --slice 20 \
    --overwrite \
    --baseline-model-type 'soap' \
    --baseline-load-file '/g/g20/vita1/ws/logs/ip_explorer/AL_Al/soap/reprs_in_arrays_representations.xyz' \
    --model-type 'mace' \
    --model-path '/g/g20/vita1/ws/projects/mace/results/ell3_corr2_int2_rmax_5_128_equi/' \
    --additional-kwargs "cutoff:5.0" \
    --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/mace/ell3_corr2_int2_rmax_5_128_equi" \
    # --vgop-kwargs "min_cut:1.0 max_cut:10.0 num_cutoffs:10 elements:['Al'] interactions:'all' pad_atoms:True" \
    # --model-load-file '/g/g20/vita1/ws/logs/ip_explorer/AL_Al/mace/ell3_corr2_int2_rmax_5_128_equi/model-representations.xyz' \
    # --baseline-load-file '/g/g20/vita1/ws/logs/ip_explorer/AL_Al/mace/ell3_corr2_int2_rmax_5_128_equi/baseline-representations.xyz' \
