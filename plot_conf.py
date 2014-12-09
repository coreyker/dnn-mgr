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

def save_conf_mat(confusion, title):
    
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
    #plt.show()

    plt.savefig(title + '.pdf', format='pdf')
    plt.close()
    return augmented_confusion[-1,-1]
    
def plot_ave_conf_mat(confusion_matrices, title):
    
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')

    labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    ave_confusion = np.mean(confusion_matrices, axis=0)
    std_confusion = np.std(confusion_matrices, axis=0)

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.imshow(np.array(ave_confusion), cmap=plt.cm.gray_r, interpolation='nearest')

    width,height = ave_confusion.shape
    for x in xrange(width):
        for y in xrange(height):
            if ave_confusion[x][y]<50:
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

    pr     = tp / (tp + fp) # precision
    rc     = tp / (tp + fn) # recall
    fscore = 2. * pr * rc / (pr + rc) # f-score
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
    
    parser.add_argument('--file', nargs='*', help='File(s) listing file name and class for each test file')
    parser.add_argument('--summary')

    args = parser.parse_args()

    # tabulate confusions
    n_classes = 10
    classes = {'blues':0, 'classical':1, 'country':2, 'disco':3, 'hiphop':4, 'jazz':5, 'metal':6, 'pop':7, 'reggae':8, 'rock':9}    
    
    ave_acc = []
    for f in args.file:
        
        confusion = np.zeros((n_classes, n_classes))
        with open(f) as fname:
            lines = fname.readlines()

        for l in lines:
            s = l.split() 
            true_class = classes[s[0].split('.')[0]]
            pred_class = int(s[2])
            confusion[pred_class, true_class] += 1

        ave = save_conf_mat(confusion, os.path.splitext(f)[0])
        ave_acc.append([f, ave])

    if args.summary:
        with open(args.summary, 'w') as f:
            for l in ave_acc:
                f.write('{}\t{}\n'.format(*l))

    # #_, model_file = sys.argv
    # model_files = ['./saved/mlp_rlu-fold-1_of_4.cpu.pkl',
    #     './saved/mlp_rlu-fold-2_of_4.cpu.pkl',
    #     './saved/mlp_rlu-fold-3_of_4.cpu.pkl',
    #     './saved/mlp_rlu-fold-4_of_4.cpu.pkl']#,
    #     #'./saved/mlp_rlu-filtered-fold.cpu.pkl']
    #     #'./saved/mlp_from_rbm-fold-1_of_4.cpu.pkl',
    #     #'./saved/mlp_from_rbm-fold-2_of_4.cpu.pkl',
    #     #'./saved/mlp_from_rbm-fold-3_of_4.cpu.pkl',
    #     #'./saved/mlp_from_rbm-fold-4_of_4.cpu.pkl',
    #     #'./saved/mlp_from_rbm-filtered-fold.cpu.pkl']

    # ave_acc = []
    # confusion_matrices = []

    # for model_file in model_files:        

    #     # get model
    #     model = serial.load(model_file)  

    #     # get dataset fold used for training from model's yaml_src
    #     p = re.compile(r"which_set.*'(train)'")
    #     dataset_yaml = p.sub("which_set: 'test'", model.dataset_yaml_src)
    #     dataset = yaml_parse.load(dataset_yaml)

    #     _, confusion = file_misclass_error(model, dataset)
        
    #     confusion = confusion.transpose() # use true columns (instead of true rows)

    #     acc = 100 * np.sum(np.diag(confusion)) / np.sum(confusion)
    #     ave_acc.append(acc)

    #     if 0:
    #         plt.ion()
    #         plot_conf_mat(confusion, title='')

    #         save_file =  os.path.splitext(os.path.splitext(model_file)[0])[0] + '.pdf'
    #         plt.savefig(save_file, format='pdf')
        
    #     confusion_matrices.append(augment_confusion_matrix(confusion))

    # # average confusions across folds
    # confusion_matrices = np.array(confusion_matrices)

    # plt.ion()
    # plot_ave_conf_mat(confusion_matrices)

    # mlp_ave = np.mean(ave_acc[:4])
    # mlp_std = np.std(ave_acc[:4])
    # mlp_filt_ave = ave_acc[4]

    # mlp_rbm_ave = np.mean(ave_acc[5:9])
    # mlp_rbm_std = np.std(ave_acc[5:9])
    # mlp_rbm_filt_ave = ave_acc[9]


