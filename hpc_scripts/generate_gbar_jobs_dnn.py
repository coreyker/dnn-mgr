import os

jobscript = '''
#!/bin/sh
# embedded options to qsub - start with #PBS
# -- Name of the job ---
#PBS -N {jobname}
# -- specify queue --
#PBS -q hpc
# -- estimated wall clock time (execution time): hh:mm:ss --
#PBS -l walltime=24:00:00
# --- number of processors/cores/nodes --
#PBS -l nodes=1:ppn=1:gpus=1
# -- user email address --
#PBS -M cmke@dtu.dk
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
cd /dtu-compute/cosound/data/_tzanetakis_genre/
python /SCRATCH/cmke/dnn-mgr/train_mlp_script.py {fold_config} {yaml_file} --nunits {nunits} --output {savename}
'''.format

fold_config = ['GTZAN_stratified.pkl']*4 + ['GTZAN_filtered.pkl']*4
yaml_file = ['mlp_rlu2.yaml', 'mlp_rlu2.yaml', 'mlp_rlu_dropout2.yaml', 'mlp_rlu_dropout2.yaml']*4
nunits = [50, 500]*8
for f, d, n in zip(fold_config, yaml_file, nunits):
	
	savename=''
	if f==fold_config[0]:
		savename += 'S_'
	else:
		savename += 'F_'

	if n==50:
		savename += '50_'
	else:
		savename += '500_'

	if d=='mlp_rlu2.yaml':
		savename += 'RS'
	else:
		savename += 'RSD'

	with open(savename+'.sh', 'w') as fname:
		fname.write(jobscript(jobname=savename, fold_config=f, yaml_file=os.path.join('/SCRATCH/cmke/dnn-mgr/',d), nunits=n, savename='/SCRATCH/cmke/saved_models/dnn/'+savename+'.pkl'))
