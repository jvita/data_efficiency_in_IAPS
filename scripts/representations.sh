#!/bin/bash
#BSUB -J "soap_debug_repr"
#BSUB -o "/usr/workspace/vita1/logs/lsf/%J.out"
#BSUB -e "/usr/workspace/vita1/logs/lsf/%J.err"
#BSUB -G c02red
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

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

# just to record each node we're using in the job output
jsrun -r 1 hostname
 
# get hostname of node that jsrun considers to be first (where rank 0 will run)
firsthost=`jsrun --nrs 1 -r 1 hostname`
echo "first host: $firsthost"

# set MASTER_ADDR to hostname of first compute node in allocation
# set MASTER_PORT to any used port number
export MASTER_ADDR=$firsthost
# export MASTER_PORT=23456

# Runtime settings
NUM_NODES=1
GPUS_PER_NODE=1
CPUS_PER_GPU=2
CPUS_PER_NODE=$(( $GPUS_PER_NODE*$CPUS_PER_GPU ))

jsrun -r 1 -c $CPUS_PER_NODE -g $GPUS_PER_NODE --bind=none python3 -m ip_explorer.representations \
    --seed 1123 \
    --num-nodes 1 \
    --gpus-per-node 1 \
    --batch-size 32 \
    --overwrite \
    --model-type "soap" \
    --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/soap" \
    --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    --additional-kwargs "n_max:10 l_max:3 cutoff:10.0 elements:['Al'] pad_atoms:False" \
    --prefix 'reprs_in_arrays_' \
    # --model-type "mace" \
    # --model-path '/g/g20/vita1/ws/projects/mace/results/ell3_corr2_int3_128_equi/' \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/debug" \
    # --additional-kwargs "cutoff:5.0" \
    # --model-type "vgop" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/vgop" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    # --additional-kwargs "min_cut:1.0 max_cut:20.0 num_cutoffs:10 elements:['Al'] interactions:'all' pad_atoms:True" \
    # --prefix 'rmax20_' \
    # --model-type 'valle-oganov' \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    # --additional-kwargs "cutoff:20 elements:['Al'] pad_atoms:False" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/valle_oganov/" \
    # --model-type "schnet" \
    # --additional-kwargs "cutoff:7.0 representation_type:node remove_offsets:False" \
    # --model-path '/g/g20/vita1/ws/logs/runs/schnet/painn/4166174-schnet_aspirin-cutoff=7.0-n_atom_basis=128-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=5000' \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/rmd17/aspirin/schnet/painn" \
    # --database-path "/g/g20/vita1/ws/projects/data/rMD17/rmd17/downsampled/aspirin" \
    # --model-type "vgop" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/rmd17/aspirin/vgop" \
    # --database-path "/g/g20/vita1/ws/projects/data/rMD17/rmd17/downsampled/aspirin" \
    # --additional-kwargs "min_cut:1.0 max_cut:7.0 num_cutoffs:10 elements:['C','H'] interactions:'all'" \
    # --additional-kwargs "min_cut:1.0 max_cut:5.0 num_cutoffs:10 elements:['C','H'] interactions:'all'" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    # --model-type "schnet" \
    # --additional-kwargs "cutoff:7.0 representation_type:node remove_offsets:False" \
    # --model-path '/g/g20/vita1/ws/logs/runs/schnet/atomwise/4165865-schnet_aspirin-cutoff=7.0-n_atom_basis=128-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=5000' \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/rmd17/aspirin/schnet/atomwise" \
    # --database-path "/g/g20/vita1/ws/projects/data/rMD17/rmd17/downsampled/aspirin" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    # --prefix 'al_using_aspirin_model_' \
    # --model-path '/g/g20/vita1/ws/projects/schnet/logs/runs/4085987-al_al_atomwise_long-cutoff=7.0-n_atom_basis=30-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=2000' \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/rmd17/aspirin/schnet/atomwise" \
    # --model-type "vgop" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/vgop" \
    # --additional-kwargs "cutoffs:[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0] elements:['Al'] interactions:'all'" \
    #--additional-kwargs "cutoffs:[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0] elements:['C','H','O','N'] interactions:'all' pad_atoms:True" \
    # --database-path "/g/g20/vita1/ws/projects/data/rMD17/rmd17/downsampled/aspirin" \
    # --model-type "nequip" \
    # --model-path "/g/g20/vita1/ws/projects/nequip/results/AL_Al/no_rescale/" \
    # --database-path "/g/g20/vita1/ws/projects/nequip/results/AL_Al/no_rescale/" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/nequip/no_rescale" \
    # --additional-kwargs "representation_type:both" \
    # --no-compute-initial-losses
