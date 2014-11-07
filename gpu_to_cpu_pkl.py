import os, sys, glob
import numpy as np
from pylearn2.utils import serial
import pylearn2.config.yaml_parse as yaml_parse
from pylearn2.models.rbm import RBM
import copy

if __name__=="__main__":

	os.environ['THEANO_FLAGS']="device=cpu"
	_, directory = sys.argv

	files_list = glob.glob(os.path.join(directory, '*.pkl'))

	for in_file in files_list:
		if in_file.split('.')[-2] == 'cpu':
			continue

		out_file = os.path.splitext(in_file)[0] + '.cpu.pkl'

		if os.path.exists(out_file):
			continue

		model  = serial.load(in_file)
		model2 = yaml_parse.load(model.yaml_src)

		params = model.get_param_values()
		model2.set_param_values(params)

		model2.yaml_src = model.yaml_src
		model2.dataset_yaml_src = model.dataset_yaml_src

		serial.save(out_file, model2)
