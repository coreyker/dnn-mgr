import sys
import numpy as np
from matplotlib import pyplot as plt

def excerpt_num(s):
	return int(s.split('.')[-2])

if __name__=='__main__':
	#_,_,fname,label = sys.argv
	cat = 2
	categories = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	pre = '/home/cmke/Development/dnn-mgr/saved_models/rf/'
	fnames = [pre+'F_500_RSD_AF_L1.txt',
		pre+'F_500_RSD_AF_L2.txt',
		pre+'F_500_RSD_AF_L3.txt',
		pre+'F_500_RSD_AF_LAll.txt']


	# get system's classifications for given label
	labels=[] # excerpt number | true label | l1 label | l2 label | l3 label | lall label 
	for fname in fnames:
		with open(fname) as f:
			a = [[excerpt_num(i), int(j), int(k)] for i,j,k in [l.split()[:3] for l in f.readlines()] if int(j)==cat]
		labels.append(np.vstack(a))
	labels = np.hstack(labels)[:,[0,1,2,5,8,11]]

	plt.figure(num=None, figsize=(9, 3), facecolor='w', edgecolor='k')
	x_range = np.arange(len(labels))
	y_range = np.arange(10)

	plt.axis([0,len(x_range),0,len(y_range)])
	ax = plt.gca()
	plt.xticks(x_range+.5, labels[:,0]) # excerpt number
	plt.yticks(y_range+.5, [c[:2] for c in categories])
	ax.yaxis.set_ticks_position('none')

	colors = [(.5,.1,.1,0.5),(.1,.5,.1,0.5),(.2,.5,.5,0.5),(.9,.1,.1,0.5)]
	offset=np.array([.125, .125/2])+np.array([[0,.5],[.5,.5],[.5,0],[0,0]])
	marker = ['1', '2', '3', 'A'] 
	width=1
	for x,lab in enumerate(labels):
		for i,l in enumerate(lab[2:]):

			r = plt.Rectangle([x,l], width, width, 
					facecolor=(.5,.5,.5,.3), edgecolor='k', linewidth=1.5)
			ax.add_patch(r)
			plt.text(x+offset[i][0],l+offset[i][1], marker[i], fontsize=9)

			#plt.plot(x+offset[i][0],l+offset[i][1],'o',color=colors[i])
	tick = plt.gca().get_yticklabels()[cat]
	tick.set_color('red')
	plt.show()
	plt.grid()
	plt.xlabel('%s excerpts' % categories[cat])
	plt.gcf().tight_layout()
	
	# width=.01
	# color=['b','g','r','k']
	# #offset=[[0,0],[width/2,0],[0,width/2],[width/2,width/2]]
	# offset=[[0,0],[width,0],[2*width,0],[3*width,0]]

	# for i,row in enumerate(a):
	# 	x = i*width*4
	# 	for j,col in enumerate(row[1:]):
	# 		y = col*width*4
	# 		r = plt.Rectangle([x+offset[j][0],y+offset[j][1]], width, width, facecolor=color[j], edgecolor='none')
	# 		ax.add_patch(r)
	# 		#plt.plot(x+offset[j][0],y+offset[j][1], 'x', color='k')

	# plt.show()
	# #plt.axis('equal')





	# ================================================================

	# plt.figure()
	# ax = plt.gca()

	# width=.01
	# color=[(1,0,0,0.25),(0,1,0,0.25),(0,0,1,0.25),(0,0.3,0.2,0.25)]
	# #offset=[[0,0],[width/2,0],[0,width/2],[width/2,width/2]]
	# offset=[[0,0],[width,0],[2*width,0],[3*width,0]]


	# # create channels
	# for i in range(10):
	# 	x=i*width*4
	# 	#for j in range(4):
	# 	#	r = plt.Rectangle([x+j*width,0], width, 1, facecolor=color[j], edgecolor='none')
	# 	#	ax.add_patch(r)
	# 	r = plt.Rectangle([x,0], 4*width, 1, facecolor=color[i%2], edgecolor='none')
	# 	ax.add_patch(r)
	# 	plt.plot((x,x),(0,1),'k-')
	# for i in range(10):
	# 	y=i*width*4
	# 	plt.plot((0,1),(y,y),'k-')
	# plt.show()
	# #plt.axis('equal')
