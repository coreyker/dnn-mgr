##About
This repository contains research code for experimenting with deep neural networks (DNNs) for music genre recognition (MGR). For the moment the repository contains many test scripts and should be considered unstable (i.e., the code is subject to change given our experimental needs). Nonetheless, the instructions below should help you reproduce our experiments and identify which files are most important for the basic functionality. 

##Requirements
- Python (Tested with version 2.7. Note Python 3 contains many changes that might introduce bugs in this code) 
- NumPy
- SciPy
- PyTables (requires numexpr and libhdf5)
- Theano
- Pylearn2
- scikits.audiolab (and libsndfile)
- scikits-learn

##V2.0 Instructions
This version is more flexible than the previous version and has been designed to work with generic datasets (not only the Tzanetakis dataset), with arbitrary categorical labels, and excerpt lengths.

###Dataset organization:
Audio files must be uncompressed in either WAV or AU format and many different types of directory structures are permissible. However, there must be a way of specifying the categorical label for each file in the dataset. This can be done either by embedding the label in the filename, or the name of the parent folder (the folder name will always take precedence in the case of a conflict).

In order to handle large datasets that may not fit into RAM this code requires that the dataset first be saved as a hdf5 file, which can be partially loaded into RAM on demand during training and testing. The script prepare_dataset2.py will search for the dataset files, and prepare the hdf5 file. Furthermore, the prepare_dataset2.py script can generate train/validation/test configuration files that specify a partition to be used in an experiment (e.g., 10-fold cross-validation). The partition configuration contains important meta-data, such as the train/valid/test files (and their index in the hdf5 file), as well as the mean and standard deviation of the training set (which can be used to standardize the data for training, validation, and testing).

The following instructions demonstrate an example of how to use the code:
####1. Prepare the dataset and partition configuration file(s):

```
python prepare_dataset2.py \
	/path/to/dataset \
	/path/to/label_list.txt \
	--hdf5 /path/to/save/dataset.hdf5 \
	--train_prop 0.5 \
	--valid_prop 0.25 \
	--test_prop 0.25 \
	--partition_name /path/to/save/partition_configuration.pkl \
	--compute_std
```

This will create the hdf5 dataset file and generate (1/test_prop) stratified partitions. The label_list.txt is a comma or newline separated list of the categorical labels in the dataset (which are matched against file and/or folder names)

Alternatively the user can use a list of files when creating the partition:

```
python prepare_dataset2.py \
	/path/to/dataset \
	/path/to/label_list.txt \
	--hdf5 /path/to/save/dataset.hdf5 \
	--train /path/to/train_list.txt \
	--valid /path/to/valid_list.txt \
	--test /path/to/test_list.txt \
	--partition_name /path/to/save/partition_configuration.pkl \
	--compute_std
```

The lists should be newline separated, and contain the relative path to each file (from the root folder of the dataset). For example if the directory structure is as follows:

/root/blues/file.wav
/root/jazz/file.wav
.
.
.

then

the training list text file might look like this:
blues/file.wav
jazz/file.wav

run: `python prepare_dataset2.py --help` to see a full list of options

####2. Train a DNN:

```
python train_mlp_script.py \
	/path/to/partition_configuration.pkl \
	/path/to/yaml_config_file.yaml \
	--nunits 50
	--output /path/to/save/model_file.pkl
```

Two yaml configuration files are provided (but you can write your own for different experiments): "mlp_rlu2.yaml" and "mlp_rlu_dropout2.yaml"

####3. Test a previously trained and saved DNN:

```
python test_mlp_script2.py \
	/path/to/saved/model_file.pkl \
	--majority_vote
```

The model knows which dataset it was trained on, and will use the associated test set. An alternative testset can also be specified:

```
python test_mlp_script2.py \
	/path/to/saved/model_file.pkl \
	--testset /path/to/alternate/partition_configuration.pkl
	--save_file /path/to/savefile.txt
```

`--save_file` lets the user save the test results to a file


##V1.0 Instructions
This version has now been removed, but can be checked out as a branch using the v1.0 tag.