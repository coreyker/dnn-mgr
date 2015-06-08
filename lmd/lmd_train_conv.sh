#!/bin/sh
# embedded options to qsub - start with #PBS
# -- Name of the job ---
#PBS -N lmd_train_conv
# -- specify queue --
#PBS -q hpc
# -- estimated wall clock time (execution time): hh:mm:ss --
#PBS -l walltime=30:00:00
# --- number of processors/cores/nodes --
#PBS -l nodes=1:ppn=1:gpus=1
# -- user email address --
#PBS -M coreyker@gmail.com
# -- mail notification --
#PBS -m abe
# -- run in the current working (submission) directory --
if test X$PBS_ENVIRONMENT = XPBS_BATCH; then cd $PBS_O_WORKDIR; fi
# here follow the commands you want to execute
# Load modules needed by myapplication.x
module load python/2.7.3 cuda/6.5

# Run my program
export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH
source ~/venv/bin/activate
cd /dtu-compute/cosound/data/_latinmusicdataset/
python ~/dnn-mgr/train_mlp_conv_script.py \
    /dtu-compute/cosound/data/_latinmusicdataset/LMD_split_conv_config.pkl \
    ~/dnn-mgr/yaml_scripts/mlp_rlu_conv2.yaml \
    --output ~/dnn-mgr/lmd/lmd_513_conv.pkl
