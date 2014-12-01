# training script
import sys, os, argparse, cPickle
import pylearn2.config.yaml_parse as yaml_parse
import pdb

if __name__=="__main__":
	
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
	description='''Script to train a 3-hidden layer DNN with a variable number of units and the possibility of using dropout.
	''')

	parser.add_argument('fold_config', help='Path to dataset partition configuration file (generated with prepare_dataset.py)')
	parser.add_argument('--nunits', type=int, help='Number of units in each hidden layer')
	parser.add_argument('--dropout', action='store_true', help='Set this flag if you want to use dropout regularization')
	parser.add_argument('--output', help='Name of output model')
	args = parser.parse_args()

	if args.nunits is None:
		parser.error('Please specify number of hidden units per layer with --nunits flag')
	if args.output is None:
		parser.error('Please specify the name that the trained model file should be saved as (.pkl file)')

	if args.dropout:
		print 'Using dropout'
		yaml_base_file = 'mlp_rlu_dropout.yaml'
	else:
		print 'Not using dropout'
		yaml_base_file = 'mlp_rlu.yaml'

	hyper_params = { 'dim_h0' : args.nunits,
		'dim_h1' : args.nunits,
		'dim_h2' : args.nunits,
		'fold_config' : args.fold_config,
		'best_model_save_path' : args.output,
		'save_path'	: '/tmp/save.pkl'
	}

	with open(yaml_base_file) as f:
		train_yaml = f.read()

	train_yaml = train_yaml % (hyper_params)
	train = yaml_parse.load(train_yaml)
	train.main_loop()
