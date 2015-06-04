#!/bin/sh
# embedded options to qsub - start with #PBS
# -- Name of the job ---
#PBS -N lmd_prep
# -- specify queue --
#PBS -q hpc
# -- estimated wall clock time (execution time): hh:mm:ss --
#PBS -l walltime=12:00:00
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
cd ~/dnn-mgr
python prepare_dataset.py \
    /dtu-compute/cosound/data/_latinmusicdataset/ \
    /dtu-compute/cosound/data/_latinmusicdataset/label_list.txt \
    --hdf5 /dtu-compute/cosound/data/_latinmusicdataset/LMD.h5 \
    --train /dtu-compute/cosound/data/_latinmusicdataset/train-part.txt \
    --valid /dtu-compute/cosound/data/_latinmusicdataset/valid-part.txt \
    --test /dtu-compute/cosound/data/_latinmusicdataset/test-part.txt \
    --partition_name /dtu-compute/cosound/data/_latinmusicdataset/LMD_split_config.pkl \
    --compute_std