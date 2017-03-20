from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import synset_to_dfs_ids
import numpy as np
from scipy.misc import imsave, imresize

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

# Solution based on the pre-trained convnets code here:
# https://github.com/heuritech/convnets-keras
# Installation instructions are at the link
# Requires Theano backend to keras

#fish_type = 'YFT'
#image_name = 'img_01938'
#image_address = currentdir + '/data/train/' + fish_type + '/' + image_name + '.jpg'


def find_fish(model_, currentdir_, fish_type_, image_name_, train_test='train'):
    if train_test == 'train':
        image_address = currentdir_ + '/data/train/' + fish_type_ + '/' + image_name_
    else:
        image_address = currentdir_ + '/data/test_stg1/' + fish_type_ + '/' + image_name_

    im = preprocess_image_batch([image_address], color_mode="rgb")

    SCALE_DOWN_IMAGE = False
    if SCALE_DOWN_IMAGE:
        scale_factor = 0.5
        im = imresize(np.squeeze(im), scale_factor)
        im = im.transpose((2, 0, 1))
        im = im[np.newaxis, :, :, :]

    out = model_.predict(im)

    # This is the label for fish
    s = "n02512053"
    ids = synset_to_dfs_ids(s)
    heatmap = out[0,ids].sum(axis=0)

    # Then, we can get the image
    SAVE_HEATMAP = False
    if SAVE_HEATMAP:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        savefile = currentdir_ + '/data/train_cropped/' + fish_type_ + image_name_ + '_hm.png'
        plt.imsave(savefile,heatmap)

    # Most likely fish location
    im_width = im.shape[3]
    im_height = im.shape[2]

    hm_width = heatmap.shape[1]
    hm_height = heatmap.shape[0]

    fishiest_point = np.unravel_index(heatmap.argmax(), heatmap.shape)

    # The heatmap size is ~33 x 16 pixels
    # Scale this up to the original image size
    tl_corner_row = int(fishiest_point[0] * (im_height - 227) / (hm_height - 1))
    tl_corner_col = int(fishiest_point[1] * (im_width - 227) / (hm_width - 1))

    output_size = 227
    br_corner_row = np.minimum(tl_corner_row + output_size, im_height)
    br_corner_col = np.minimum(tl_corner_col + output_size, im_width)

    # Crop the image and save
    cropped = im[:,:,tl_corner_row:br_corner_row,tl_corner_col:br_corner_col]
    cropped = np.squeeze(cropped, axis=0)
    cropped = cropped.transpose((1, 2, 0))
    if train_test == 'train':
        savefile = currentdir_ + '/data/train_cropped/' + fish_type_ + '/' + image_name_[:-4] + '.png'
    else:
        savefile = currentdir_ + '/data/test_cropped/' + image_name_[:-4] + '.png'
    imsave(savefile,cropped)


# Need to download the weights and save them in a directory
# called 'weights'
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
weights_path = currentdir + '/weights/alexnet_weights.h5'
model = convnet('alexnet',weights_path=weights_path, heatmap=True)
model.compile(optimizer=sgd, loss='mse')

categories = ['ALB', 'BET', 'DOL', 'LAG', 'NoG', 'OTHER', 'SHARK', 'YFT']

for i, c in enumerate(categories):

    print('Processing category: ' + str(c))

    # Directory where the source images are, for each category
    rootdir = './data/train/{}'.format(categories[i])

    # Directory where the cropped images will be saved, for each category
    newdir = './data/train_cropped/{}'.format(categories[i])

    for fish_image_name in os.listdir(rootdir):
        if not fish_image_name.startswith('.'):
            find_fish(model, currentdir, c, fish_image_name, 'train')

# test images
testdir = './data/test_stg1'
for fish_image_name in os.listdir(testdir):
    if not fish_image_name.startswith('.'):
        find_fish(model, currentdir, c, fish_image_name, 'test')



