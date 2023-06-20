#!/bin/bash
#SBATCH --job-name "vo_pes_MOLECULE"
#SBATCH --output "/usr/workspace/vita1/logs/lsf/%J.out"
#SBATCH --error "/usr/workspace/vita1/logs/lsf/%J.err"
#SBATCH --account c02red
#SBATCH --partition pbatch
#SBATCH -N 1
#SBATCH -t 00:30:00

# Environment setup
module load gcc/8.3.1
module load cuda/11.6.1

# conda init bash
# conda activate conda-development
source /g/g20/vita1/venv-ruby/bin/activate

# Used for PyTorch Lightning
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR=$(hostname)

LOG_DIR='/g/g20/vita1/ws/logs/ip_explorer/AL_Al/vgop'
PREFIX=''
DIMENSIONS=2

sheap -v -hs -rs -1 -p 20 -st 1.0 -dim $DIMENSIONS < "${LOG_DIR}/${PREFIX}representations.xyz" > "${LOG_DIR}/${PREFIX}sheap-${DIMENSIONS}d.xyz"
# sheap -v -scale -hs -p 20 -st 0.6 -dim $DIMENSIONS < "${LOG_DIR}/${PREFIX}representations.xyz" > "${LOG_DIR}/${PREFIX}sheap-${DIMENSIONS}d.xyz"

python3 -m ip_explorer.pes \
    --load-dir ${LOG_DIR} \
    --prefix ${PREFIX} \
    --n-components ${DIMENSIONS} \
    --scale 0.1 \
