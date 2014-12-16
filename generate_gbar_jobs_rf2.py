import os

jobscript = '''
#!/bin/sh
# embedded options to qsub - start with #PBS
# -- Name of the job ---
#PBS -N {jobname}
# -- specify queue --
#PBS -q hpc
# -- estimated wall clock time (execution time): hh:mm:ss --
#PBS -l walltime=12:00:00
# --- number of processors/cores/nodes --
#PBS -l nodes=1:ppn=4:gpus=1
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

cd /dtu-compute/cosound/data/_tzanetakis_genre
python /SCRATCH/cmke/dnn-mgr/train_classifier_on_dnn_feats2.py {model_file} --which_layers {which_layers} --save_file {save_file} {aggregate_features}
'''.format

model_files  = ['S_50_RS.pkl', 'S_50_RSD.pkl', 'S_500_RS.pkl', 'S_500_RSD.pkl', 'F_50_RS.pkl', 'F_50_RSD.pkl', 'F_500_RS.pkl', 'F_500_RSD.pkl']
dataset_dir  = '/dtu-compute/cosound/data/_tzanetakis_genre/audio'
which_layers = ['1', '2', '3', '1 2 3']
aggregate_features = ['--aggregate_features', '']

job_list = []
for agg in aggregate_features:
	for model in model_files:
		for l in which_layers:

			savename = model.split('.pkl')[0]
			if agg=='':
				savename += '_FF_' # frame-level features
			else:
				savename += '_AF_' # aggregate features

			if l=='1 2 3':
				savename += 'LAll'
			else:
				savename += 'L' + l

			jobname = savename+'.sh'
			job_list.append( jobname )
			with open(jobname, 'w') as fname:
				fname.write( jobscript( jobname=savename, 
										model_file=os.path.join('/SCRATCH/cmke/saved_models/dnn', model), 
										which_layers=l, 
										save_file=os.path.join('/SCRATCH/cmke/saved_models/rf', savename), 
										aggregate_features=agg) )

with open('_master_RF_trainer.sh', 'w') as f:
	for j in job_list:
		f.write('qsub %s\n' % j)


