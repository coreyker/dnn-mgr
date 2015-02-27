import os, sys, tables
import numpy as np

def create_split_files(hdf5, ntrain, nvalid, ntest, path):
    # ntrain, nvalid, ntest are per class

    # extract metadata from dataset
    hdf5_file = tables.open_file(hdf5, mode='r')
    param     = hdf5_file.get_node('/', 'Param')    
    file_dict = param.file_dict[0]
    
    train_list = []
    valid_list = []
    test_list  = []

    rng = np.random.RandomState(111)
    for key, files in file_dict.iteritems(): # for all files that share a given label
        nfiles = len(files)             
        perm = rng.permutation(nfiles)      

        sup         = np.arange(ntest)
        train_index = perm[:ntrain]
        valid_index = perm[ntrain:ntrain+nvalid]
        test_index  = perm[ntrain+nvalid:ntrain+nvalid+ntest]

        train_list.append([files[i] for i in train_index])
        valid_list.append([files[i] for i in valid_index])
        test_list.append([files[i] for i in test_index])

    # flatten lists
    train_list = sum(train_list,[])
    valid_list = sum(valid_list,[])
    test_list  = sum(test_list,[])

    with open(os.path.join(path, 'train-part.txt'), 'w') as f:
        for i in train_list:        
            f.write('{}\n'.format(i))

    with open(os.path.join(path, 'valid-part.txt'), 'w') as f:
        for i in valid_list:        
            f.write('{}\n'.format(i))

    with open(os.path.join(path, 'test-part.txt'), 'w') as f:
        for i in test_list:        
            f.write('{}\n'.format(i))

    hdf5_file.close()

if __name__=='__main__':
    create_split_files(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])