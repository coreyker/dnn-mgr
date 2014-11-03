import os, sys, glob
from pylearn2.utils import serial
import pylearn2.config.yaml_parse as yaml_parse

if __name__=="__main__":

	os.environ['THEANO_FLAGS']="device=cpu"
	_, directory = sys.argv

	for in_file in glob.glob(os.path.join(directory + '*.pkl')):
		if in_file.split('.')[-2] == 'cpu':
			continue
		
		out_file = os.path.splitext(d)[0] + '.cpu.pkl'
		
		if os.path.exists(out_file):
			continue

		model = serial.load(in_file)
	 
		model2 = yaml_parse.load(model.yaml_src)
		model2.set_param_values(model.get_param_values())
	 
		serial.save(out_file, model2)