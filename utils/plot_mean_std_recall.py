import os, glob
import numpy as np
from matplotlib import pyplot as plt

def compute_recall(fname, n_classes=10):
    
    with open(fname) as f:
        lines = [l.split() for l in f.readlines()]
    
    confusion = np.zeros((n_classes, n_classes))
    for (fname, true_label, pred_label) in lines:
        confusion[int(pred_label), int(true_label)] += 1

    confusion /= np.sum(confusion, axis=0)
    recalls = np.diag(confusion)

    return np.mean(recalls), np.std(recalls)/np.sqrt(250)*2.145

def get_freq_from_fname(f):
    fname = os.path.splitext(os.path.split(f)[-1])[0]
    freq = fname.split('-')[-1]
    return float(freq)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description='')
    parser.add_argument('--dnn_dir')
    parser.add_argument('--aux_dir')

    args = parser.parse_args()

    dnn_files = glob.glob(os.path.join(args.dnn_dir, '*.txt'))
    aux_files = glob.glob(os.path.join(args.aux_dir, '*.txt'))

    dnn_recall=[]
    for d in dnn_files:
        f = get_freq_from_fname(d)
        m,s = compute_recall(d)
        dnn_recall.append((f,m,s))
    dnn_recall = np.vstack(dnn_recall)
    sortind = np.argsort(dnn_recall[:,0])
    dnn_recall = np.vstack([dnn_recall[i,:] for i in sortind])

    aux_recall=[]
    for a in aux_files:
        f = get_freq_from_fname(a)
        m,s = compute_recall(a)
        aux_recall.append((f,m,s))
    aux_recall = np.vstack(aux_recall)
    sortind = np.argsort(aux_recall[:,0])
    aux_recall = np.vstack([aux_recall[i,:] for i in sortind])

    plt.ion()
    plt.figure()
    color1=[1,0.6,0.1]
    color2=[.4,.6,1]
    #plt.errorbar(dnn_recall[:,0], dnn_recall[:,1], yerr=dnn_recall[:,2], fmt='o')
    plt.plot(dnn_recall[:,0], dnn_recall[:,1], 'o-', color=tuple(color1+[0.8]), linewidth=2)
    plt.fill_between(dnn_recall[:,0], dnn_recall[:,1]-dnn_recall[:,2], dnn_recall[:,1]+dnn_recall[:,2], color=tuple(color1+[0.4]))

    #plt.errorbar(aux_recall[:,0], aux_recall[:,1], yerr=aux_recall[:,2], fmt='o')
    plt.plot(aux_recall[:,0], aux_recall[:,1], 'o-', color=tuple(color2+[0.8]), linewidth=2)
    plt.fill_between(aux_recall[:,0], aux_recall[:,1]-aux_recall[:,2], aux_recall[:,1]+aux_recall[:,2], color=tuple(color2+[0.3]))

    plt.grid()
    plt.xlabel('Lowpass cutoff frequency (Hz)')
    plt.ylabel('Recall')
    plt.axis('tight')
    plt.axis([20, 11000, 0, 1])

    #plt.savefig('recall_vs_cutoff.pdf', format='pdf')