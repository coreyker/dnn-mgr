# training script
import sys, os, cPickle
import pylearn2.config.yaml_parse as yaml_parse
import pdb

if __name__=="__main__":
	
	fold_config = sys.argv[1] # e.g., GTZAN_1024-fold-1_of_4.pkl
	yaml_base_file = sys.argv[2] # e.g., mlp_rlu.yaml

	with open(fold_config) as f:
		cfg = cPickle.load(f)

	base  = cfg['h5_file_name'].split('.h5')[0]
	ext   = fold_config.split(base)[1]
	model = yaml_base_file.split('.yaml')[0] + ext

	hyper_params = {
		'fold_config' : fold_config,
		'best_model_save_path' : './saved/' + model,
		'save_path'	: './saved/save.pkl'
	}

	with open(yaml_base_file) as f:
		train_yaml = f.read()

	train_yaml = train_yaml % (hyper_params)
	train = yaml_parse.load(train_yaml)
	train.main_loop()
