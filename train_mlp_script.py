# training script
import os
import pylearn2.config.yaml_parse as yaml_parse


yaml_base_file = 'mlp_rlu.yaml'
ext = '-fold-4_of_4.pkl'

hyper_params = { 'dim_h0' : 50,
	'dim_h1' : 50,
	'dim_h2' : 50,
	'fold_config' : 'GTZAN_1024' + ext,
	'best_model_save_path' : './saved-rlu-505050/mlp_rlu' + ext,
	'save_path'	: './saved-rlu-505050/save.pkl'
}

with open(yaml_base_file) as f:
	train_yaml = f.read()

train_yaml = train_yaml % (hyper_params)
train = yaml_parse.load(train_yaml)
train.main_loop()
