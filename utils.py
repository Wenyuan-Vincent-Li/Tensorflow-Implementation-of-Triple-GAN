import sys
import os
if os.getcwd().endswith("Path_Semi_GAN"):
    root_dir = os.getcwd()
else:
    root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.misc

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.client import device_lib

def files_under_folder_with_suffix(dir_name, suffix = ''):
    """
    Return a filename list that under certain folder with suffix
    :param dir_name: folder specified
    :param suffix: suffix of the file name, eg '.jpg'
    :return: List of filenames in order
    """
    files = [f for f in os.listdir(dir_name) if (os.path.isfile(os.path.join(dir_name, f)) and f.endswith(suffix))]
    files.sort()
    return files

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_sementic(image, segmentation_mask, num_classes=int(4),
                     title="", figsize=(16, 16), ax=None):
    label_colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]

    ## 0/stroma : red; 1/low-grade: green; 2/high-grade: blue 3/benign: yellow

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    masked_image = image.astype(np.uint32).copy()

    for label in range(num_classes):
        mask = np.zeros_like(segmentation_mask)
        mask[np.where(segmentation_mask == label)] = 1
        masked_image = apply_mask(masked_image, mask, label_colours[label])

    ax.imshow(masked_image.astype(np.uint8))
    plt.show()

def plot(samples,Nh,Nc,channel,IMG_HEIGHT, IMG_WIDTH):
    fig = plt.figure(figsize=(Nc, Nh))
    plt.clf()
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i in range(samples.shape[0]):
        sample = samples[i, :]
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
            immin=(image[:,:]).min()
            immax=(image[:,:]).max()
            image=(image-immin)/(immax-immin+1e-8)
            plt.imshow(image,cmap ='gray')
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
            immin=(image[:,:,:]).min()
            immax=(image[:,:,:]).max()
            image=(image-immin)/(immax-immin+1e-8)
            plt.imshow(image)
    return fig

def check_folder_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return False
    else:
        return True

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def variable_name_string():
    name_string = ''
    for v in tf.global_variables():
        name_string += v.name + '\n'
    return name_string


def get_trainable_weight_num(var_list=tf.trainable_variables()):
    total_parameters = 0
    for variable in var_list:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def variable_name_string_specified(variables):
    name_string = ''
    for v in variables:
        name_string += v.name + '\n'
    return name_string


def grads_dict(gradients, histogram_dict):
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        histogram_dict[variable.name + "/gradients"] = gfrad_values
        histogram_dict[variable.name + "/gradients_norm"] = \
            clip_ops.global_norm([grad_values])
    return histogram_dict


def fn_inspect_checkpoint(ckpt_filepath, **kwargs):
    name = kwargs.get('tensor_name', '')
    if name == '':
        all_tensors = True
    else:
        all_tensors = False
    chkp.print_tensors_in_checkpoint_file(ckpt_filepath, name, all_tensors)


def save_dict_as_txt(Dict, dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    for key in Dict.keys():
        np.savetxt(dir_name + key + '.txt', np.asarray(Dict[key]))


def convert_list_2_nparray(varlist):
    var_np = np.empty((0,))
    for i in range(len(varlist)):
        var_np = np.concatenate((var_np, varlist[i]))
    return var_np


def add_grads(grads, histogram_dict):
    for index, grad in enumerate(grads):
        tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])


def initialize_uninitialized_vars(sess):
    from itertools import compress
    global_vars = tf.global_variables()
    local_vars = tf.local_variables()
    init_var = global_vars + local_vars
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in init_var])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

    local_vars = tf.local_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in local_vars])
    not_initialized_vars = list(compress(local_vars, is_not_initialized))

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
    return (images + 1.)/2.


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
          i = idx % size[1]
          j = idx // size[1]
          img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


###########################
## Test code
###########################
def _main_variable_name_string():
    print(variable_name_string())


def _main_inspect_checkpoint(save_dir):
    from Training.Saver import Saver
    saver = Saver(save_dir)
    _, filename, _ = saver._findfilename()
    fn_inspect_checkpoint(filename)

if __name__ == "__main__":
    ckpt_file_path = os.path.join(os.getcwd(), "Training/Weight_mnist/Run_2019-03-19_17_40_57/model_1000.ckpt")
    fn_inspect_checkpoint(ckpt_file_path,tensor_name = "")