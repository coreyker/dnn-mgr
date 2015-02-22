# training script
import sys, os, argparse, cPickle
import pylearn2.config.yaml_parse as yaml_parse
import pdb

if __name__=="__main__":
	
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
	description='''Script to train a DNN with a variable number of units and the possibility of using dropout.
	''')

	parser.add_argument('fold_config', help='Path to dataset partition configuration file (generated with prepare_dataset.py)')
	parser.add_argument('yaml_file')
	parser.add_argument('--output', help='Name of output model')
	args = parser.parse_args()

	if args.output is None:
		parser.error('Please specify the name that the trained model file should be saved as (.pkl file)')

	hyper_params = { 
		'fold_config' : args.fold_config,
		'best_model_save_path' : args.output,
		'save_path'	: '/tmp/save.pkl'
	}

	with open(args.yaml_file) as f:
		train_yaml = f.read()

	train_yaml = train_yaml % (hyper_params)
	train = yaml_parse.load(train_yaml)
	train.main_loop()