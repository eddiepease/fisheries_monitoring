import numpy as np
from scipy import misc
from sklearn_theano.feature_extraction import OverfeatLocalizer
import os
from sklearn.mixture import GMM
# from skimage.transform import resize


def convert_gmm_to_box(gmm, width = 3):
    """
    Function to convert a fitted gmm into bounding box coordinates.
    It takes the parts of the image that are thought to contain
    fish and fits a gaussian over the top.  We then consider a
    bounding box centered at the mean of the gaussian with size
    related to the standard deviation of the gaussian.

    std is a hyperparameter.  Higher -> larger bounding box.

    :param gmm: A GMM that has been fitted to the fish points.

    :param gmm: Size of bounding box.

    :return: x and y max/min points of area of image that contain
             fish.
    """
    midpoint = gmm.means_             # Centre of bounding box
    std = width * np.sqrt(gmm.covars_)
    width = std[:, 0]                 # Width of bounding box
    height = std[:, 1]                # Height of bounding box

    # cast to ints because they will be used as pixel indices
    ymin = int(midpoint[:, 0] - width // 2)
    ymax = int(midpoint[:, 0] + width // 2)
    xmin = int(midpoint[:, 1] - height // 2)
    xmax = int(midpoint[:, 1] + height // 2)

    return xmin, xmax, ymin, ymax


def calculate_bounding_box(xmin, xmax, ymin, ymax, nrows, ncols):
    """
    Function that takes as arguments candidate bounding box dimensions
    and an image size, and returns actual (realistic) bounding
    box dimensions.  The calculated bounding box is at least
    100x100 pixels and is within the image itself.

    :param xmin: Candidate xmin
    :param xmax: Candidate xmax
    :param ymin: Candidate ymin
    :param ymax: Candidate ymax
    :param nrows: Number of rows of image (2nd dimension in np array)
    :param ncols: Number of cols of image (1st dimension in np array)
    :return: The four bounding box dimensions
    """

    # Make sure the bounding boxes are at least 100x100 pixels
    if (xmax - xmin) < 100:
        current_size = xmax - xmin
        xmax = int(xmax + (100 - current_size) / 2)
        xmin = int(xmin - (100 - current_size) / 2)

    if (ymax - ymin) < 100:
        current_size = ymax - ymin
        ymax = int(ymax + (100 - current_size) / 2)
        ymin = int(ymin - (100 - current_size) / 2)

    # In order to preserve our bounding box size, if our candidate
    # bounding box is outside the image, we shift it to be inside the image
    if xmin < 0:
        xmax = xmax - xmin  # xmin is negative, so this adds to xmax
        xmin = 0

    if ymin < 0:
        ymax = ymax - ymin  # xmin is negative, so this adds to xmax
        ymin = 0

    if xmax > nrows:
        xmin = xmin + nrows - xmax - 1  # xmax is bigger than nrows, so this reduces xmin
        xmax = nrows - 1

    if ymax > ncols:
        ymin = ymin + nrows - ymax - 1  # ymax is bigger than nrows, so this reduces ymin
        ymax = ncols - 1

    # Final idiot filter
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)

    xmax = min(xmax, nrows - 1)
    ymax = min(ymax, ncols - 1)

    return xmin, xmax, ymin, ymax


##########################################################
# Rough structure
# For each image:
# - Finding the fishy bits of the image
# - Using a GMM to fit a gaussian over the fishy points,
#   to help with robustness to outliers.
# - Calculate a realistic bounding box (i.e. inside image,
#   not too small.
# - Crop the image to the bounding box size.
# - Save image.
#
##########################################################

# This code assumes that the fish images are in the usual
# place: ./data/train
#
# AND that there is a destination folder for the cropped
# fish images: ./data/train_cropped   with similar
# folders for the categories.

FULL_DATASET = True  # Set to false to use different directories. For testing.

if FULL_DATASET:
    categories = ['ALB', 'BET', 'DOL', 'LAG', 'NoG', 'OTHER', 'SHARK', 'YFT']
else:
    categories = ['ALB', 'SHARK']


for i, c in enumerate(categories):

    print('Processing category: ' + str(c))

    if FULL_DATASET:
        # Directory where the source images are, for each category
        rootdir = './data/train/{}'.format(categories[i])

        # Directory where the cropped images will be saved, for each category
        newdir = './data/train_cropped/{}'.format(categories[i])

    else:
        # Directory where the source images are, for each category
        rootdir = './data/train_small/{}'.format(categories[i])

        # Directory where the cropped images will be saved, for each category
        newdir = './data/train_cropped_small/{}'.format(categories[i])

    for fish_image_name in os.listdir(rootdir):

        if not fish_image_name.startswith('.'):    # stops hidden files causing a problem

            print('Processing image: ' + str(fish_image_name))

            # Read the fish image into 3D numpy array
            fish_image = misc.imread(os.path.join(rootdir, fish_image_name))

            # Get all points with fish in the top label.  NB these are not pixels, but
            # seem to be a grid over the image.
            fish_label = 'fish.n.01'
            clf = OverfeatLocalizer(match_strings=[fish_label], top_n=1)  # top_n is hyperparameter

            bounding_box_gmm_size = 3  # can change gmm selection as iterations proceed

            # [n x 2] numpy array containing grid points (not every pixel) that
            # is fishy.
            fish_points = clf.predict(fish_image)

            if fish_points:   # i.e. list of fish points is not empty because there are none

                # Get the number of fishy points.  This is important because if there
                # are e.g. 1 or none then we can't fit a GMM.

                num_fish_points, _ = np.shape(fish_points[0])

                if num_fish_points > 5:

                    my_gmm = GMM()
                    my_gmm.fit(fish_points[0])
                    xmin, xmax, ymin, ymax = convert_gmm_to_box(my_gmm, width=bounding_box_gmm_size)

                    # For this first iteration we want to have some extra pixels because
                    # the bounding boxes tend to cut bits of fish off.

                    xmin -= 30
                    ymin -= 30
                    xmax += 30
                    ymax += 30

                    # Prevent indexing outside the image.
                    # Get image size
                    nrows, ncols, nchannels = np.shape(fish_image)

                    xmin, xmax, ymin, ymax = calculate_bounding_box(xmin, xmax, ymin, ymax, nrows, ncols)

                    # Fill in the bits beyond the bounding box with black
                    #fish_image[:xmin, :, :] = 0.
                    #fish_image[xmax:, :, :] = 0.
                    #fish_image[:, :ymin, :] = 0.
                    #fish_image[:, ymax:, :] = 0.

            # save image, cropped to the size of the final bounding box
            fish_image_cropped = fish_image[xmin:xmax+1, ymin:ymax+1, :]

            misc.imsave(os.path.join(newdir, fish_image_name), fish_image_cropped)
