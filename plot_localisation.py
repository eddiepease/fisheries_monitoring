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
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
from matplotlib.patches import Rectangle



def convert_points_to_box_coord(points):

    wmin = points[:, 0].min()
    wmax = points[:, 0].max()
    hmin = points[:, 1].min()
    hmax = points[:, 1].max()

    return wmin, wmax, hmin, hmax


def convert_gmm_to_box(gmm):
    midpoint = gmm.means_
    std = 3 * np.sqrt(gmm.covars_)
    width = std[:, 0]
    height = std[:, 1]

    ymin = int(midpoint[:, 0] - width // 2)  # int because they are pixel numbers
    ymax = int(midpoint[:, 0] + width // 2)
    xmin = int(midpoint[:, 1] - height // 2)
    xmax = int(midpoint[:, 1] + height // 2)

    return xmin, xmax, ymin, ymax


categories = ['ALB', 'BET', 'DOL', 'LAG', 'NoG', 'OTHER', 'SHARK', 'YFT']

for i, c in enumerate(categories):

    print(c)
    #rootdir = './data/train_small_source/{}'.format(categories[i])
    rootdir = './data/train_small_cropped/{}'.format(categories[i])

    # newdir = './data/train_small_cropped/{}'.format(categories[i])
    newdir = './data/train_small_cropped2/{}'.format(categories[i])

    for fish in os.listdir(rootdir):
    #for fish in os.scandir(rootdir):
        if not fish.startswith('.'):  # and fish.is_file():   # stops hidden files causing a problem

            print(fish)

            X = misc.imread(os.path.join(rootdir, fish))
            #X = misc.imread(fish)
            # print(1)

            # Get all points with fish in the top 5 labels
            fish_label = 'fish.n.01'
            clf = OverfeatLocalizer(match_strings=[fish_label], top_n=1)

            # for j in range(5):
            fish_points = clf.predict(X)[0]

            # colour all the fish points with black blobs
            #for (y, x) in zip(fish_points[:, 0], fish_points[:, 1]):   # NB x / y order
            #    X[x-2:x+2, y-2:y+2, :] = 0.


            #wmin, wmax, hmin, hmax = convert_points_to_box_coord(fish_points)

            # print(X.shape)
            # print(hmin, hmax, wmin, wmax)

            #cropped = X[hmin:hmax,wmin:wmax,:]

            # fish = 'b_{}'.format(fish)

            # Draw the box on the image
            num_fish_points, _ = np.shape(fish_points)

            if num_fish_points > 5:

                my_gmm = GMM()
                my_gmm.fit(fish_points)
                xmin, xmax, ymin, ymax = convert_gmm_to_box(my_gmm)

                nrows, ncols, nchannels = np.shape(X)

                # Prevent indexing outside the image
                xmin = max(xmin, 0)
                ymin = max(xmin, 0)
                xmax = min(xmax, ncols)
                ymax = min(ymax, nrows)

                # paint out all the non-fish bits in black
                X[:xmin, :, :] = 0.
                X[xmax:, :, :] = 0.
                X[:, :ymin, :] = 0.
                X[:, ymax:, :] = 0.

            misc.imsave(os.path.join(newdir, fish), X)
            #misc.imsave(fish, X)


            # print('saved')

    # resized = resize(cropped, (128, 128, 3))
    # scipy.misc.imsave('./train/outfile1.jpg', resized)



    # play around











