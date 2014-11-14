#About
This repository contains research code evaluating deep neural networks (DNNs) for music genre recognition (MGR).

#Requirements:
Python as well as the following python packages (and their dependencies):
NumPy
SciPy
PyTables (requires numexpr and libhdf5)
Theano
Pylearn2
scikits.audiolab (and libsndfile)

#Instructions:
1. Download the Tzanetakis genre set http://opihi.cs.uvic.ca/sound/genres.tar.gz
2. Copy all of the files into a single directory
3. Convert the files to WAV format 
4. Clone the repository to a local machine: git clone https://github.com/coreyker/dnn-mgr.git
5. Cd into the reprository directory
6. run: python prepare_dataset.py /path/to/tzanetakis/wav/files
7. run: python train_mlp_script.py GTZAN_1024-fold-1_of_4.pkl mlp_rlu.yaml
8. run: python test_mlp_script.py ./saved/mlp_rlu-fold-1_of_4.pkl


