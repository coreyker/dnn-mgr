from matplotlib import pyplot as plt

def plot_conf_mat(conf, ax_labels):
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	ax.set_aspect(1)
	ax.imshow(np.array(conf), cmap=plt.cm.gray_r, interpolation='nearest')

	width,height = conf.shape
	for x in xrange(width):
	    for y in xrange(height):
	    	if x==y:
	    		color='w'
	    	else:
	    		color='k'
	        ax.annotate('%2.2f'%conf[x][y], xy=(y, x), horizontalalignment='center', verticalalignment='center',color=color)

	ax.xaxis.tick_top()
	plt.xticks(range(width), ax_labels)
	plt.yticks(range(height), ax_labels)

	labels = ax.get_xticklabels() 
	for label in labels: 
	    label.set_rotation(30) 

	plt.show()

# plot confusions error
fold_err  = []
fold_conf = []

run test_mlp_script.py GTZAN_1024-fold-1_of_4.pkl ./saved/mlp_rlu-fold-1_of_4.pkl
fold_err.append(err)
fold_conf.append(conf)

run test_mlp_script.py GTZAN_1024-fold-2_of_4.pkl ./saved/mlp_rlu-fold-2_of_4.pkl
fold_err.append(err)
fold_conf.append(conf)

run test_mlp_script.py GTZAN_1024-fold-3_of_4.pkl ./saved/mlp_rlu-fold-3_of_4.pkl
fold_err.append(err)
fold_conf.append(conf)

run test_mlp_script.py GTZAN_1024-fold-4_of_4.pkl ./saved/mlp_rlu-fold-4_of_4.pkl
fold_err.append(err)
fold_conf.append(conf)

fold_conf = np.array(fold_conf, dtype=np.float32)

ave_err = np.mean(fold_err)
std_err = np.std(fold_err)

ave_conf = np.mean(fold_conf/25.*100, axis=0)
std_conf = np.std(fold_conf/25.*100, axis=0)

ax_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
plot_conf_mat(ave_conf, ax_labels)
#plot_conf_mat(ave_conf, ax_labels)

run test_mlp_script.py GTZAN_1024-filtered-fold.pkl ./saved/mlp_rlu-filtered-fold.pkl
filt_err = err
filt_conf = conf
ave_filt_conf = filt_conf/np.sum(filt_conf, axis=1) * 100
plot_conf_mat(ave_filt_conf, ax_labels)

