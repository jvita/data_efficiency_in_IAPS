#!/bin/bash
#BSUB -J "painn_3BPA_test_1200K_err"
#BSUB -o "/usr/workspace/vita1/logs/lsf/%J.out"
#BSUB -e "/usr/workspace/vita1/logs/lsf/%J.err"
#BSUB -G c02red
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 1:00
#BSUB -alloc_flags ipisolate

# Environment setup
module load gcc/8.3.1
module load cuda/11.6.1

# Activate conda environment
source /usr/workspace/vita1/programs/anaconda/bin/activate
conda activate opence-1.7.2-cuda-11.4

# Used for PyTorch Lightning
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# just to record each node we're using in the job output
jsrun -r 1 hostname
 
# get hostname of node that jsrun considers to be first (where rank 0 will run)
firsthost=`jsrun --nrs 1 -r 1 hostname`
echo "first host: $firsthost"

# set MASTER_ADDR to hostname of first compute node in allocation
# set MASTER_PORT to any used port number
export MASTER_ADDR=$firsthost
export MASTER_PORT=23556

# Runtime settings
NUM_NODES=1
GPUS_PER_NODE=1
CPUS_PER_GPU=1
CPUS_PER_NODE=$(( $GPUS_PER_NODE*$CPUS_PER_GPU ))

# -r: number of resource sets per node
# -a: number of "tasks" per resource set (default is 1 task = 1 process)
# -c: number of CPUs per resource set
# -g: number of GPUs per resource set
# --bind=none: allow each task to use all of its allocated cpus
# jsrun -r 1 -a $GPUS_PER_NODE -c $CPUS_PER_NODE -g $GPUS_PER_NODE --bind=none python3 -m ip_explorer.landscape \
jsrun --smpiargs='off' -r 1 -a $GPUS_PER_NODE -c $CPUS_PER_NODE -g $GPUS_PER_NODE --bind=none python3 -m ip_explorer.landscape \
    --num-nodes $NUM_NODES \
    --gpus-per-node $GPUS_PER_NODE \
    --batch-size 1 \
    --loss-type 'both' \
    --landscape-type 'plane' \
    --steps 41 \
    --distance 2.0 \
    --overwrite \
    --no-compute-landscape \
    --model-type 'painn' \
    --model-path '/usr/workspace/vita1/logs/runs/3BPA/painn/4268450-painn_3bpa_1000Wf-cutoff=5.0-n_atom_basis=256-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=1.0-Fw=1000.0-lr=0.005-epochs=5000' \
    --database-path '/usr/workspace/vita1/projects/data/3BPA/dataset_3BPA/'\
    --save-dir "/g/g20/vita1/ws/logs/ip_explorer/3BPA/painn/baseline" \
    --additional-kwargs "cutoff:5.0" \
    --additional-datamodule-kwargs "cutoff:5.0 train_filename:/g/g20/vita1/ws/projects/data/3BPA/dataset_3BPA/test_1200K.xyz" \
    # --model-type 'nequip' \
    # --model-path '/g/g20/vita1/ws/projects/nequip/results/AL_Al/baseline_10Wf' \
    # --database-path '/g/g20/vita1/ws/projects/nequip/results/AL_Al/baseline_10Wf' \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/nequip/baseline_redo" \
    # --model-type 'nequip' \
    # --model-path '/g/g20/vita1/ws/projects/nequip/results/3BPA/1000_Wf' \
    # --database-path '/g/g20/vita1/ws/projects/nequip/results/3BPA/1000_Wf' \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/benzene/nequip/1000_Wf" \
    # --n-lines 20 \
    # --additional-datamodule-kwargs "train_filename:/g/g20/vita1/ws/projects/data/3BPA/dataset_3BPA/test_1200K.xyz" \
    # --model-type 'mace' \
    # --model-path '/g/g20/vita1/ws/projects/mace/results/3BPA/updated_arch2/mace_3_2_2_1_128_rar_yesblockres_yesmodelres/' \
    # --additional-kwargs "cutoff:5.0" \
    # --additional-datamodule-kwargs "cutoff:5.0 train_filename:/g/g20/vita1/ws/projects/data/3BPA/dataset_3BPA/test_1200K.xyz" \
    # --database-path "/g/g20/vita1/ws/projects/data/3BPA/dataset_3BPA" \
    # --save-dir '/g/g20/vita1/ws/logs/ip_explorer/3BPA/mace/mace_3_2_2_1_128_rar_yesblockres_yesmodelres_highres' \
    # --no-compute-initial-losses \
    # --additional-datamodule-kwargs "cutoff:5.0 train_filename:/g/g20/vita1/ws/projects/data/AL_Al" \
    # --model-type 'painn' \
    # --model-path '/g/g20/vita1/ws/logs/runs/painn/4121168-painn_lr_sched-cutoff=7.0-n_atom_basis=128-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=5000' \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/painn/best" \
    # --additional-kwargs "cutoff:7.0" \
    # --prefix 'bugfix_' \
    # --no-compute-initial-losses \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/mace/ell0_corr2_int2_rmax_5_128_inv" \
    # --model-type 'painn' \
    # --model-path '/g/g20/vita1/ws/projects/painn/logs/runs/4085987-al_al_atomwise_long-cutoff=7.0-n_atom_basis=30-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=2000/' \
    # --additional-kwargs "cutoff:7.0" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/painn/atomwise" \
    # --seed 1123 \
    # --model-type "painn" \
    # --model-path '/g/g20/vita1/ws/logs/runs/painn/painn/4166174-painn_benzene-cutoff=7.0-n_atom_basis=128-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=5000' \
    # --database-path "/g/g20/vita1/ws/projects/data/rMD17/rmd17/downsampled/benzene" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/rmd17/benzene/painn/atomwise" \
    # --additional-kwargs "cutoff:7.0" \
    # --database-path "/g/g20/vita1/ws/projects/data/rMD17/rmd17/downsampled/benzene" \
    # --model-type "allegro" \
    # --model-path "/g/g20/vita1/ws/projects/allegro/results/AL_Al/initial/" \
    # --database-path "/g/g20/vita1/ws/projects/allegro/results/AL_Al/initial/" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/allegro/initial" \
    # --model-type "mace" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    # --model-path '/g/g20/vita1/ws/projects/mace/corr3_128/checkpoints/'\
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/mace/corr3_equi" \
    # --additional-kwargs "cutoff:5.0" \
    # --model-type "test" \
    # --database-path "test" \
    # --model-path 'test' \
    # --save-dir "test" \
    # --model-type "ace" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    # --model-path '/g/g20/vita1/ws/projects/ace/senary' \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/ace/senary" \
    # --additional-kwargs "cutoff:7.0" \
    # --model-type "painn" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    # --model-path '/g/g20/vita1/ws/logs/runs/painn/4121168-painn_lr_sched-cutoff=7.0-n_atom_basis=128-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=5000' \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/painn/painn" \
    # --additional-kwargs "cutoff:7.0" \
    # --overwrite \
    # --layers-regex "skip_tp" \
