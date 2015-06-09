import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

import sys, re, os
import numpy as np
from pylearn2.utils import serial
import pylearn2.config.yaml_parse as yaml_parse

#from test_mlp_script import frame_misclass_error, file_misclass_error

def plot_conf_mat(confusion, title, labels):        
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
            ax.annotate('%2.1f'%augmented_confusion[x][y], xy=(y, x), horizontalalignment='center', verticalalignment='center',color=color, fontsize=9)

    ax.xaxis.tick_top()
    plt.xticks(range(width), labels+['Pr'])
    plt.yticks(range(height), labels+['F'])

    xlabels = ax.get_xticklabels()
    for label in xlabels: 
        label.set_rotation(30) 

    plt.xlabel(title)
    plt.show()

def save_conf_mat(confusion, title, labels):
    augmented_confusion = augment_confusion_matrix(confusion)

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.imshow(np.array(augmented_confusion), cmap=plt.cm.gray_r, interpolation='nearest')

    thresh = np.max(augmented_confusion)
    width,height = augmented_confusion.shape
    for x in xrange(width):
        for y in xrange(height):
            if augmented_confusion[x][y]<thresh/2:
                color='k'
            else:
                color='w'
            ax.annotate('%2.1f'%augmented_confusion[x][y], xy=(y, x), horizontalalignment='center', verticalalignment='center',color=color, fontsize=9)

    ax.xaxis.tick_top()
    plt.xticks(range(width), labels+['Pr'])
    plt.yticks(range(height), labels+['F'])

    xlabels = ax.get_xticklabels()
    for label in xlabels: 
        label.set_rotation(30) 

    #plt.xlabel(title.split('/')[-1])
    #plt.show()

    plt.savefig(title + '.pdf', format='pdf')
    plt.close()
    return augmented_confusion[-1,-1]
    
def plot_ave_conf_mat(confusion_matrices, title, labels):
    
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')

    ave_confusion = np.mean(confusion_matrices, axis=0)
    std_confusion = np.std(confusion_matrices, axis=0)

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.imshow(np.array(ave_confusion), cmap=plt.cm.gray_r, interpolation='nearest')

    thresh = np.max(ave_confusion)
    width,height = ave_confusion.shape
    for x in xrange(width):
        for y in xrange(height):
            if ave_confusion[x][y]<thresh/2:#50:
                color='k'
            else:
                color='w'
            mean = ave_confusion[x][y]
            std  = std_confusion[x][y]
            ax.annotate('$%2.1f$' % mean, xy=(y, x), horizontalalignment='center', verticalalignment='bottom',color=color, fontsize=11)
            ax.annotate('$\pm %2.1f$' % std, xy=(y, x), horizontalalignment='center', verticalalignment='top',color=color, fontsize=9)

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

    pr     = tp / ((tp + fp) + 1e-12) # precision
    rc     = tp / ((tp + fn) + 1e-12)# recall
    fscore = 2. * pr * rc / ((pr + rc) + 1e-12) # f-score
    ave    = np.sum(np.diag(confusion)) / np.sum(confusion) * 100

    confusion = confusion / np.sum(confusion, axis=0) * 100 # precentages
    augmented_confusion = np.hstack((confusion, 100 * np.reshape(pr, (len(pr),1))))
    augmented_confusion = np.vstack((augmented_confusion, np.hstack((100 * fscore, ave))))

    return augmented_confusion

if __name__ == '__main__':

    #ex1: python plot_conf.py --file ./saved/*.txt
    #ex2: python plot_conf.py --summary ./saved/RF_500/RF_500_summary.txt --file `find ./saved/RF_500/ -name "*.txt" -print`
    import argparse
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description='''Script to generate confusion matrices.
        ''')
    
    parser.add_argument('--file', nargs='*', help='Tab separated file(s) listing filename, true class, and predicted class')
    parser.add_argument('--labels', help='Text file with list of labels')
    parser.add_argument('--summary')

    args = parser.parse_args()

    # tabulate confusions
    with open(args.labels) as f:
        lines = f.readlines()
        if len(lines)==1: # assume comma separated, single line
            label_list = lines[0].replace(' ','').split(',')
        else:
            label_list = [l.split()[0] for l in lines]


    ave_acc = []
    for f in args.file:
                
        with open(f) as fname:
            lines = fname.readlines()

        mx = np.max([int(l.strip().split('\t')[-2]) for l in lines])
        mn = np.min([int(l.strip().split('\t')[-2]) for l in lines])
        n_classes = mx-mn+1

        confusion = np.zeros((n_classes, n_classes))

        for l in lines:
            s = l.strip().split('\t') 
            true_class = int(s[1])-mn #classes[s[0].split('.')[0]]
            pred_class = int(s[2])-mn
            confusion[pred_class, true_class] += 1

        ave = save_conf_mat(confusion, os.path.splitext(f)[0], label_list)
        ave_acc.append([f, ave])

    if args.summary:
        with open(args.summary, 'w') as f:
            for l in ave_acc:
                f.write('{}\t{}\n'.format(*l))


