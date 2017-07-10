from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import re
import operator
import glob
import csv
import math
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import linalg as la
from scipy.special import betainc
# from matplotlib import pyplot as plt
import time
import flock

import scipy.ndimage.filters as fi
import scipy.stats as st
import scipy.misc
import multiprocessing

# Make paths absolute and independent from where the python script is called.
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

sys.path.append(os.path.join(script_dir, '..', 'sealionengine'))

from sealiondata import SeaLionData

#
# Setup
#

dry_run = False               # Disables file writing for debugging purporses.
cls_nb = 5                    # Number of classes.
scale_factor = 0.60           # Scale factor by which the training data scaled.
tile_size = 256               # Size of the tiles.
tile_margin = 64              # How much context (overlap) the tile should include.
coords_margin = 128           # Margin within which coordinates should be included.
aug_factor = 1.0              # Factor by which the most numerous class should be augmented.
background_rate = 0.02        # How often to include tiles without animals.
pca_samples_per_example = 3   # Number of pixels to sample per training example.
aug_threshold = 3             # Below # of animals images are not augmented to save space.
aug_max = 1.                  # Maximal number of augmentations of all parameters per example.
aug_min = 0.                  # Minimal number of augmentations of all parameters per example.
aug_frac = 0.6                # Fraction of augmentations of all parameters per example.
masked_region_threshold = 40  # Below this threshold pixels are taken from TrainDotted.
num_processes = 8             # Number of processes to process the data in parallel.
storage_split_nb = 100        # Number of files that the training data should be split into.
clear_files = True            # Whether to override the training data.
recompute_coords = False      # Whether to recompute the coordinates of sealionengine.
test_color_aug = False        # Whether to run testing code for color augmentation.
train_val_split = 0.1         # How to split the data set.
train_suffix = 'train'        # Name the of the training data tfrecords.
val_suffix = 'val'            # Name the of the validation data tfrecords.

data_dir = os.path.join(script_dir)
input_dir = os.path.join(script_dir, '..', 'input')
debug_dir = os.path.join(script_dir, '..', 'debug')
pca_file = os.path.join(data_dir, 'pca')

tf.gfile.MakeDirs(debug_dir)

coords_file = os.path.join(script_dir, '../sealionengine/outdir/coords.csv')

sld = SeaLionData(input_dir, data_dir)

cls_sizes = (
  61,  # adult_males
  51,  # subadult_males
  51,  # adult_females
  31,  # juveniles
  21   # pubs
)
max_cls_size = max(cls_sizes)

# First augmentation is the identity.
aug_rot = (0, 1, 2, 3)
aug_scale = (1.0, 1. / 1.1, 1.1)
aug_flip = (False, True)

aug_params = [(rot, scale, flip) for rot in aug_rot \
                                 for scale in aug_scale \
                                 for flip in aug_flip]


#
# Functions
#

def applyColorAugmentation(img, std=0.5):
  '''Applies random color augmentation following [1].

  [1] Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. \
    ImageNet Classification with Deep Convolutional Neural Networks. \
    NIPS 2012.'''

  alpha = np.clip(np.random.normal(0, std, size=3), -2 * std, 2. * std)
  perturbation = sld_evecs.dot((alpha * np.sqrt(sld_evals)).T)
  gamma = 1.0 - sum(perturbation) / 3.
  return np.power(np.clip(img + perturbation, 0., 1.), gamma)
  return np.clip((img + perturbation), 0., 1.)


def getTransformedTileAndCoords(img, ref_y, ref_x, coords=None, \
                                rot=0, scale=1., flip=False):
  
  src_size = round(scale * actual_tile_size + 2 * actual_tile_margin)
  half_size = src_size / 2.
  margin = (src_size - actual_tile_size) // 2
  y = ref_y - margin
  x = ref_x - margin
  dst_size = tile_size + 2 * tile_margin
  coords_pad = actual_coords_margin - actual_tile_margin

  if y + img_pad < 0 or y + src_size + img_pad < 0 \
    or y + img_pad > img.shape[0] or y + src_size + img_pad > img.shape[0] \
    or x + img_pad < 0 or x + src_size + img_pad < 0 \
    or x + img_pad > img.shape[1] or x + src_size + img_pad > img.shape[1]:

    raise ValueError('Indices out of bounds: %i, %i, %i, %i' %
                     (y + img_pad, y + src_size + img_pad, x + img_pad, x + src_size + img_pad))

  tile = cv2.resize(img[y + img_pad:y + src_size + img_pad, \
                        x + img_pad:x + src_size + img_pad], \
                    (dst_size, dst_size), \
                    interpolation=cv2.INTER_LANCZOS4) 
                  # interpolation=cv2.INTER_AREA)

  if coords is not None:
    # Needs a deep copy because now we are going to transform the coordinates.
    coords = getCoordsInSquareWithMargin(coords, y, x, src_size, coords_pad).copy()

    corr_scale = float(dst_size - 1) / float(src_size - 1)

    # print(coords)
    coords.loc[:, 'row'] = corr_scale * (coords.loc[:, 'row'] - y - half_size)
    coords.loc[:, 'col'] = corr_scale * (coords.loc[:, 'col'] - x - half_size)
    # print(coords)
    
    if flip:
      tile = tile[::-1]
      coords.loc[:, 'row'] = -coords.loc[:, 'row']

    if rot != 0:
      tile = np.rot90(tile, rot)
      
      M_rot = None
      if rot == 1: M_rot = np.asarray([[ 0., -1.], [ 1.,  0.]])
      if rot == 2: M_rot = np.asarray([[-1.,  0.], [ 0., -1.]])
      if rot == 3: M_rot = np.asarray([[ 0.,  1.], [-1.,  0.]])

      coords.loc[:, ['row', 'col']] = np.dot(M_rot, coords.loc[:, ['row', 'col']].T).T

    coords.loc[:, 'row'] += corr_scale * half_size
    coords.loc[:, 'col'] += corr_scale * half_size
    coords = np.round(coords)
    
    if dry_run:
      imshow(tile, coords=coords, save=True, title='Transformed')

  if dry_run:
    tile_data = b''
  else:
    tile_data = cv2.imencode('.png', np.asarray(tile)[..., ::-1].astype(np.uint8))[1].tostring()
    
  return tile_data, coords, tile


def countCoords(coords):
  return np.asarray([len(coords.loc[coords['cls'] == cls_idx])
                     for cls_idx in range(cls_nb)], dtype=np.int32)


def compHistDistance(h1, h2):
  def normalize(h):    
    if np.sum(h) == 0: 
        return h
    else:
        return h / np.sum(h)

  def smoothstep(x, x_min=0., x_max=1., k=2.):
      m = 1. / (x_max - x_min)
      b = - m * x_min
      x = m * x + b
      return betainc(k, k, np.clip(x, 0., 1.))

  def fn(X, Y, k):
    return 4. * (1. - smoothstep(Y, 0, (1 - Y) * X + Y + .1)) \
      * np.sqrt(2 * X) * smoothstep(X, 0., 1. / k, 2) \
             + 2. * smoothstep(Y, 0, (1 - Y) * X + Y + .1) \
             * (1. - 2. * np.sqrt(2 * X) * smoothstep(X, 0., 1. / k, 2) - 0.5)

  h1 = normalize(h1)
  h2 = normalize(h2)

  return max(0, np.sum(fn(h2, h1, len(h1))))
  # return np.sum(np.where(h2 != 0, h2 * np.log10(h2 / (h1 + 1e-10)), 0))  # KL divergence


def getCoordsInSquareWithMargin(coords, y, x, size, margin):
  return coords.loc[(coords['row'] >= y - margin)
                    & (coords['row'] < y + size + margin) \
                    & (coords['col'] >= x - margin)
                    & (coords['col'] < x + size + margin)]


def generateGaussianKernel(size):
  kernel = np.zeros((size, size))
  kernel[size // 2, size // 2] = 1.
  gauss = fi.gaussian_filter(kernel, size // 2 // 3)
  gauss[gauss < gauss[0, size // 2]] = 0.
  return gauss


cls_gaussians = [[generateGaussianKernel(math.floor(size * scale * scale_factor)) \
                  for size in cls_sizes] \
                 for scale in aug_scale]


imshow_counter = 0
def imshow(img, coords=None, title='Image', wait=True, destroy=True, save=False, normalize=False):
  global imshow_counter

  img = img.copy().astype(np.float32)

  def fill_region(dst, y, x, h, w, v):
    h2 = h // 2
    w2 = w // 2
    py = y - h2 
    px = x - w2 
    y_min = max(0, py)
    y_max = min(dst.shape[0], y + h2)
    x_min = max(0, px)
    x_max = min(dst.shape[1], x + w2)
    if y_max - y_min <= 0 or x_max - x_min <= 0:
      return

    dst[y_min:y_max, x_min:x_max] = v

  if normalize:
    img -= np.min(img)
    m = np.max(img)
    if m != 0.:
      img /= m

  if save:
    if np.all(img <= 1.0):
      img *= 255.
      
  if coords is not None:
    img = np.copy(img)
    if isinstance(coords, pd.DataFrame):
      for coord in coords.itertuples():
        fill_region(img, int(round(coord.row)) - 2, int(round(coord.col)) - 2, \
                    5, 5, np.asarray(sld.cls_colors[coord.cls]))
    else:
      for coord in coords:
        fill_region(img, int(round(coord[1])) - 2, int(round(coord[2])) - 2, \
                    5, 5, np.asarray(sld.cls_colors[coord[0]]))

  if save:
    if len(img.shape) == 2:
      img = img[:, :, None]
    lockfile = os.path.join(debug_dir, '.lock')
    with open(lockfile, 'w') as fp:
        with flock.Flock(fp, flock.LOCK_EX) as lock:
          curr_num = len(os.listdir(debug_dir))
          filename = os.path.join(debug_dir, 'imshow_%s_%i.png' % (title, curr_num)) 
          cv2.imwrite(filename, img[..., ::-1])

    return

  plt.title(title)
  plt.imshow(img)
  plt.show()

  if wait:
    input('Press enter to continue...')
  if destroy:
    plt.close()

    
if recompute_coords or not os.path.exists(coords_file):
  sld.save_coords()

  # Error analysis
  sld.verbosity = VERBOSITY.VERBOSE
  tid_counts = sld.count_coords(sld.tid_coords)
  rmse, frac = sld.rmse(tid_counts)

  print('\nRMSE: %f' % rmse)
  print('Error frac: %f' % frac)

sld_coords = pd.read_csv(coords_file)
cls_counts = [len(sld_coords.loc[sld_coords['cls'] == cls_idx]) for cls_idx in range(cls_nb)]

actual_tile_size = round(tile_size * (1. / scale_factor))
actual_tile_margin = round(tile_margin * (1. / scale_factor))
actual_coords_margin = round(coords_margin * (1. / scale_factor))
img_pad = math.ceil(((max(aug_scale) * actual_tile_size + 2 *
                      actual_tile_margin + 1) - actual_tile_size) / 2.)


def storeExampleRoundRobin(img_data, coords, scale=1.):
  global curr_writer, writers, writer_indices

  if coords is not None:
    coords_data = coords[['cls', 'row', 'col']].as_matrix().reshape((-1)).astype(np.int64)    
    coords_data_len = coords.shape[0]
  else:
    coords_data = np.empty((0), dtype=np.int64)
    coords_data_len = 0

  dst_size = tile_size + 2 * tile_margin
  feature_dict = {
    'image/height':  tf.train.Feature(int64_list=tf.train.Int64List(value=[dst_size])),
    'image/width':   tf.train.Feature(int64_list=tf.train.Int64List(value=[dst_size])),
    'image/scale':   tf.train.Feature(float_list=tf.train.FloatList(value=[scale])),
    'image':         tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_data])),
    'coords/length': tf.train.Feature(int64_list=tf.train.Int64List(value=[coords_data_len])),
    'coords':        tf.train.Feature(int64_list=tf.train.Int64List(value=coords_data))
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  if not dry_run:
    writers[writer_indices[curr_writer]].write(example.SerializeToString())
  curr_writer += 1
  if curr_writer >= len(writers):
    np.random.shuffle(writer_indices) 
    curr_writer = 0


def processExample(tid):
  np.random.seed(tid)
  
  coords = sld_coords[sld_coords['tid'] == tid]
  img = sld._load_image('train', tid, 1., 0)
  shape = img.shape
  dot_img = sld.load_dotted_image(tid, 1., 0)
  dot_img_sum = dot_img.astype(np.uint16).sum(axis=-1)[..., None]
  img = np.where(dot_img_sum < masked_region_threshold, dot_img, img)
  del dot_img, dot_img_sum
  img = np.pad(img, ((img_pad, img_pad), (img_pad, img_pad), (0, 0)),
               mode='constant')

  def get_sample(img): 
    sample_positions = np.random.randint(0, img.shape[0], size=(2, pca_samples_per_example))
    return img[sample_positions[0], sample_positions[1]]

  aug_indices = list(range(1, len(aug_params)))
  tile_counter = 0
  background_tile_counter = 0
  class_counts = np.zeros((cls_nb,), dtype=np.int64)
  pca_samples = []
  dist_list = []
  examples = []
  y_count = math.floor(shape[0] / actual_tile_size)
  x_count = math.floor(shape[1] / actual_tile_size)
  start_y = (shape[0] - actual_tile_size * y_count) // 2
  start_x = (shape[1] - actual_tile_size * x_count) // 2
  for y_idx in range(y_count):
    for x_idx in range(x_count):
      y = start_y + y_idx * actual_tile_size
      x = start_x + x_idx * actual_tile_size
      coords_ = getCoordsInSquareWithMargin(coords, y, x, actual_tile_size, 0)
      counts = countCoords(coords_)

      if np.all(np.asarray(counts) == 0):
        if np.random.random() <= background_rate:
          example = getTransformedTileAndCoords(img, y, x)
          examples.append(example[:2])  # Only append the encoded tile.
          pca_sample = get_sample(example[2])  # We sample from the raw tile.
          pca_samples.append(pca_sample)
          background_tile_counter += 1
          tile_counter += 1
      else:
        # Data balancing (commented out because it did not improve the model) 
        # if sum(counts) < 3:
        #   aug_frac = 0.
        #   # Extra boost for adult_male and subadult_male.
        #   if counts[1] > 0:
        #     aug_frac = 0.4
        #   if counts[0] > 0:
        #     aug_frac = 0.55
        # else:
        #   dist = compHistDistance(counts, cls_counts)
        #   dist_list.append(dist)
        #   aug_frac = aug_max * (1. - np.exp(-np.max(dist - 2.5, 0) ** 2. / 7.5)) + aug_min

        # # If many cls 3, reduce aug_frac a little (currently 67K, should be ~30K).
        # aug_frac *= np.exp(-np.max(counts[3] - 0., 0) ** 2 / 300.)
        # # If many cls 4, increase aug_frac a little (currently 15K, should be ~30K).
        # aug_frac = np.clip(aug_frac + 0.14 * (1. - np.exp(-np.max(counts[4] - 0., 0) ** 2 \
        #                                                  / 300.)), 0., 1.)
        
        np.random.shuffle(aug_indices)
        for aug_idx in [0] + aug_indices[:math.floor(aug_frac * len(aug_params))]:
          example = getTransformedTileAndCoords(img, y, x, coords, *aug_params[aug_idx])
          examples.append(example[:2])
          pca_sample = get_sample(example[2])
          pca_samples.append(pca_sample)
          coords_ = getCoordsInSquareWithMargin(example[1], 0, 0, tile_size, -tile_margin)
          class_counts += countCoords(coords_)
          tile_counter += 1

  return tile_counter, background_tile_counter, class_counts, pca_samples, examples, dist_list


def create_set(train_ids, split_nb, suffix, save_pca=False, shall_clear_files=False):
  global curr_writer, writers, writer_indices

  pool = multiprocessing.Pool(processes=num_processes)

  sess = tf.Session()
  
  # Prepare workspace.
  curr_writer = 0
  writer_indices = list(range(split_nb))
  writers = []
  if not dry_run:
    if clear_files and shall_clear_files:
      filenames = tf.gfile.Glob(os.path.join(data_dir, '*_%s.tfrecords' % suffix))
      for filename in filenames:
        tf.gfile.Remove(filename)
      print('All *_%s.tfrecords have been deleted.' % suffix)
  for i in range(split_nb):
    filename = os.path.join(data_dir, 'data_%03i_%s.tfrecords' % (i, suffix))
    writers.append(tf.python_io.TFRecordWriter(filename))

  tile_counter = 0
  background_tile_counter = 0
  class_counts = np.zeros((cls_nb,))
  pca_samples = []
  dist_list = []
  
  print('Generating training set...')
  image_counter = 0
  prev_time = None
  curr_rate = 280.
  start_time = time.perf_counter()
  for tile_counter_, background_tile_counter_, class_counts_, pca_samples_, examples_, dist_list_ \
     in pool.imap_unordered(processExample, train_ids):

    tile_counter += tile_counter_
    background_tile_counter += background_tile_counter_
    for i in range(len(class_counts)):
       class_counts[i] += class_counts_[i]
    pca_samples += pca_samples_
    dist_list += dist_list_
    image_counter += 1
    for example in examples_:
      storeExampleRoundRobin(*example)

    if prev_time is None:
      prev_time = time.perf_counter()
    else:
      curr_time = time.perf_counter()
      elapsed = curr_time - prev_time
      curr_rate = (29. * curr_rate + (60. / elapsed)) / 30.
      prev_time = curr_time
    time_str = time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start_time))
    eta_str = time.strftime('%H:%M:%S', time.gmtime(60. * (len(train_ids) - image_counter) \
                                                    / curr_rate))

    print('Progress: %3.1f%%, Tiles: %i, Histogram: %s, Images/Min.: %3.1f, Elapsed: %s, ETA: %s' \
          % (float(image_counter) / float(len(train_ids)) * 100, \
                         tile_counter, str(class_counts), curr_rate, time_str, eta_str))
  
  print('')
  
  pool.close()
  pool.join()

  print('Computing PCA...')

  data = np.vstack(pca_samples).astype(np.float32) / 255.
  mean = data.mean(axis=0)
  data -= mean
  cov = np.cov(data, rowvar=False)
  evals, evecs = la.eigh(cov)
  if save_pca:
    np.savez(pca_file, evecs=evecs, evals=evals, mean=mean, cov=cov)

  print('-' * 61)
  print('Final report:')
  print('-' * 61)
  print('Number of processed images:')
  print(len(train_ids))
  print('Number of tiles:')
  print(tile_counter)
  print('Number of background tiles:')
  print(background_tile_counter)
  print('Class histogram:')
  print(class_counts)
  print('Normalized class histogram:')
  print(class_counts / (np.sum(class_counts) + 1e-10))
  print('Normalized class histogram (overall):')
  print(cls_counts / (np.sum(cls_counts) + 1e-10))
  print('Histogram of class distances:')
  print(np.histogram(np.asarray(dist_list), bins=32))
  print('Number of PCA samples:')
  print(data.shape[0])
  print('Mean:')
  print(mean)
  print('Covariance:')
  print(cov)
  print('Eigenvalues:')
  print(evals)
  print('Eigenvectors:')
  print(evecs)
  print('-' * 61)

  
def main(argv):
  global sld_evecs, sld_evals
  
  # Test color augmentation
  if test_color_aug:
    if not os.path.exists(pca_file + '.npz'):
      print('Failed to test data augmentation: PCA file does not exist.')
      sys.exit(0)

    pca = np.load(pca_file + '.npz')
    sld_evecs = pca['evecs']
    sld_evals = pca['evals']
    sld_mean = pca['mean']
    sld_cov = pca['cov']

    im = sld._load_image('train', 41, 10, 0) / 255.
    for i in range(100):
      scipy.misc.imsave(os.path.join(debug_dir, 'color_aug_test_%i.png' % i), \
                        applyColorAugmentation(im) * 255.)
    sys.exit(0)

  train_ids = np.asarray(sld.train_ids)
  split_index = round(train_val_split * len(train_ids))
  shuffled_ids = list(range(len(train_ids)))
  np.random.shuffle(shuffled_ids)
  train_split_nb = round((1. - train_val_split) * storage_split_nb)
  val_split_nb = round(train_val_split * storage_split_nb)
  shall_clear_files = input('Delete all .tfrecords? (y/N) ').lower() == 'y'
  create_set(train_ids[shuffled_ids[split_index:]], train_split_nb, suffix=train_suffix,
             save_pca=True, shall_clear_files=shall_clear_files)
  create_set(train_ids[shuffled_ids[:split_index]], val_split_nb, suffix=val_suffix,
             save_pca=False, shall_clear_files=shall_clear_files)

  print('train_ids=', repr(train_ids[shuffled_ids[split_index:]]))
  print('validation_ids=', repr(train_ids[shuffled_ids[:split_index]]))


if __name__ == '__main__':
    main(sys.argv)
