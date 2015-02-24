#!/bin/bash

#python prepare_dataset.py \
#        ../GTZAN \
#        ./label_list_GTZAN.txt \
#        --hdf5 ./GTZAN.hdf5 \
#	--nfft 4096 --nhop 2048 \
#        --train ./gtzan/train_stratified2.txt \
#        --valid ./gtzan/valid_stratified2.txt \
#        --test ./gtzan/test_stratified2.txt \
#        --partition_name ./GTZANstrat_partition_configuration.pkl

python train_mlp_script.py \
        ./GTZANstrat_partition_configuration.pkl \
        ./yaml_scripts/mlp_rlu2.yaml \
        --nunits 50
        --output GTZAN_strat2049_model.pkl
