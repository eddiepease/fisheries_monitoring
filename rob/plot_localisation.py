print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
import sys
print(sys.version)

#import scipy.misc
from scipy import misc

from matplotlib.patches import Rectangle
from sklearn_theano.datasets import load_sample_image
from sklearn_theano.feature_extraction import OverfeatLocalizer
from sklearn_theano.feature_extraction import get_all_overfeat_labels

from skimage.io import imread,imsave
from skimage.transform import resize
import os



def convert_points_to_box_coord(points):
  wmin = points[:, 0].min()
  wmax = points[:, 0].max()
  hmin = points[:, 1].min()
  hmax = points[:, 1].max()
  return wmin,wmax,hmin,hmax

categories = ['ALB','BET','DOL','LAG','NoG','OTHER','SHARK','YFT']

for i, c in enumerate(categories):
	print(c)
	rootdir = '../data/train/{}'.format(categories[i])
	newdir = './train/{}'.format(categories[i])
	for fish in os.listdir(rootdir):
		print(fish)

		X = misc.imread(os.path.join(rootdir, fish))
		#print(1)

		# Get all points with fish in the top 5 labels
		fish_label = 'fish.n.01'
		clf = OverfeatLocalizer(match_strings=[fish_label])

		#for j in range(5):
		fish_points = clf.predict(X)[0]

		wmin,wmax,hmin,hmax = convert_points_to_box_coord(fish_points)
		#print(X.shape)
		#print(hmin,hmax,wmin,wmax)
		cropped = X[hmin:hmax,wmin:wmax,:]
		#fish = 'b_{}'.format(fish)
		misc.imsave(os.path.join(newdir, fish), cropped)
		#print('saved')

	#resized = resize(cropped, (128, 128, 3))
	#scipy.misc.imsave('./train/outfile1.jpg', resized)

