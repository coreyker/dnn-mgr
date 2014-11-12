import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

import sys, re, os
import numpy as np
from pylearn2.utils import serial
import pylearn2.config.yaml_parse as yaml_parse

from test_mlp_script import frame_misclass_error, file_misclass_error

def plot_conf_mat(confusion, title):
    
    labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    augmented_confusion = augment_confusion_matrix(confusion)

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.imshow(np.array(augmented_confusion), cmap=plt.cm.gray_r, interpolation='nearest')

    width,height = augmented_confusion.shape
    for x in xrange(width):
        for y in xrange(height):
            if augmented_confusion[x][y]<50:
                color='k'
            else:
                color='w'
            ax.annotate('%2.1f'%augmented_confusion[x][y], xy=(y, x), horizontalalignment='center', verticalalignment='center',color=color)

    ax.xaxis.tick_top()
    plt.xticks(range(width), labels+['Pr'])
    plt.yticks(range(height), labels+['F'])

    xlabels = ax.get_xticklabels()
    for label in xlabels: 
        label.set_rotation(30) 

    plt.xlabel(title)
    plt.show()


def augment_confusion_matrix(confusion):
    # add precision, f-score and average to confusion matrix
    # confusion: confusion matrix with true columns
    
    tp = np.diag(confusion) # true positive count
    fp = np.sum(confusion, axis=1) - tp # false positive count
    fn = np.sum(confusion, axis=0) - tp # false negative count

    pr     = tp / (tp + fp) # precision
    rc     = tp / (tp + fn) # recall
    fscore = 2. * pr * rc / (pr + rc) # f-score
    ave    = np.sum(np.diag(confusion)) / np.sum(confusion) * 100

    confusion = confusion / np.sum(confusion, axis=0) * 100 # precentages
    augmented_confusion = np.hstack((confusion, 100 * np.reshape(pr, (len(pr),1))))
    augmented_confusion = np.vstack((augmented_confusion, np.hstack((100 * fscore, ave))))

    return augmented_confusion

if __name__ == '__main__':
    
    #_, model_file = sys.argv
    model_files = ['./saved/mlp_rlu-fold-1_of_4.cpu.pkl',
        './saved/mlp_rlu-fold-2_of_4.cpu.pkl',
        './saved/mlp_rlu-fold-3_of_4.cpu.pkl',
        './saved/mlp_rlu-fold-4_of_4.cpu.pkl',
        './saved/mlp_rlu-filtered-fold.cpu.pkl',
        './saved/mlp_from_rbm-fold-1_of_4.cpu.pkl',
        './saved/mlp_from_rbm-fold-2_of_4.cpu.pkl',
        './saved/mlp_from_rbm-fold-3_of_4.cpu.pkl',
        './saved/mlp_from_rbm-fold-4_of_4.cpu.pkl',
        './saved/mlp_from_rbm-filtered-fold.cpu.pkl']

    ave_acc = []
    for model_file in model_files:        

        # get model
        model = serial.load(model_file)  

        # get dataset fold used for training from model's yaml_src
        p = re.compile(r"which_set.*'(train)'")
        dataset_yaml = p.sub("which_set: 'test'", model.dataset_yaml_src)
        dataset = yaml_parse.load(dataset_yaml)

        _, confusion = file_misclass_error(model, dataset)
        
        confusion = confusion.transpose() # use true columns (instead of true rows)

        acc = 100 * np.sum(np.diag(confusion)) / np.sum(confusion)
        ave_acc.append(acc)

        plt.ion()
        plot_conf_mat(confusion, title='')

        save_file =  os.path.splitext(os.path.splitext(model_file)[0])[0] + '.pdf'
        plt.savefig(save_file, format='pdf')

    mlp_ave = np.mean(ave_acc[:4])
    mlp_std = np.std(ave_acc[:4])
    mlp_filt_ave = ave_acc[4]

    mlp_rbm_ave = np.mean(ave_acc[5:9])
    mlp_rbm_std = np.std(ave_acc[5:9])
    mlp_rbm_filt_ave = ave_acc[9]


