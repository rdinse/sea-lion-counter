import sys
import os, copy
import tensorflow as tf
import numpy as np

import models.utilities as utilities

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'tf_models', 'slim'))

import tensorflow.contrib.slim as slim
from nets import inception_v4

from models import BasicModel

class InceptionModel(BasicModel):

  def get_default_config(self):
    config = BasicModel.get_default_config(self)
    model_config = {
      'stride': 1,
      'inception_v4_checkpoint_file': os.path.join(script_dir, '..',
                                                   'data', 'inception_v4.ckpt'),
      'batch_norm_decay': 0.99,
      'batch_norm_epsilon': 0.001,
      'output_size': 29,
      'pad': 32,
      'receptive_field_size': 66,
      'projective_field_size': 7,
      'contextual_pad': 32,
      'normalize_inputs': False,
      'batch_size': 64,
    }
    config.update(model_config)
    return config


  @staticmethod
  def get_random_hps(fixed_params={}):
    config = {
      'batch_size': np.random.choice([8, 16, 32, 64])
    }
    config.update(fixed_params)
    return config
  

  def init_variables(self):
    inception_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/InceptionV4')
    inception_vars_map = { var.name[var.name.find('InceptionV4'):].split(':')[0]: var \
                           for var in inception_vars }

    restore_saver = tf.train.Saver(var_list=inception_vars_map)
    restore_saver.restore(self.sess, self.config['inception_v4_checkpoint_file'])

    return inception_vars


  def build_model_graph(self, inputs, mode):
    image_batch, target_batch, tid_batch = inputs

    # The tiles are extended by a margin of 32 px. This roughly corresponds to
    # the extend by which the receptive field in Mixed_5d are . This way, when
    # the window slides toward the boundaries of the image, the extended
    # receptive field can recognize the boundaries early enough such that it
    # can correctly make the distinction whether the center of an animal is
    # inside or outside of the tile.

    with tf.name_scope('model') as scope:
      pad = self.config['receptive_field_size'] - self.config['contextual_pad']
      image_batch = tf.pad(image_batch, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
      batch_norm_params = {
        'decay': self.config['batch_norm_decay'],
        'epsilon': self.config['batch_norm_epsilon'],
        'is_training': (mode == self.MODE.TRAIN),
      }
      with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=(mode == self.MODE.TRAIN)):
          net, end_points = inception_v4.inception_v4_base(image_batch, final_endpoint='Mixed_5d')
      
      preds = slim.conv2d(net, self.config['cls_nb'], [1, 1], scope='Conv2d_1x1_preds', activation_fn=None)

    if mode == self.MODE.TRAIN:
      # Build target cache
      self.target_sizes = np.empty((self.config['tile_size'], self.config['tile_size'], 4), dtype=np.int32)
      input_shape = [dim.value for dim in image_batch.get_shape()]
      max_size= 0
      output_size = 0
      for y in range(self.config['tile_size']):
        input_r = utilities.Rect(y + self.config['receptive_field_size'], 0,
                                 y + self.config['receptive_field_size'], 0,
                                 input_shape[1], input_shape[2])
        r = utilities.calc_projective_field(image_batch.name, end_points['Mixed_5a'].name, input_r)
        output_size = r.h
        if r.height > max_size:
          max_size = r.height
        self.target_sizes[y, :, 0] = r.min_y
        self.target_sizes[:, y, 1] = r.min_y
        self.target_sizes[y, :, 2] = r.max_y + 1  # For Python ranges and indices
        self.target_sizes[:, y, 3] = r.max_y + 1  # which exclude the upper bound.

      self.config['projective_field_size'] = max_size

      if self.config['debug']:
        print('Projective field size: %i' % max_size)
        print('Output size: %i' % output_size)

      # Make all projective fields the same size.
      for y in range(self.config['tile_size']):
        for x in range(self.config['tile_size']):
          if self.target_sizes[y, x, 2] - self.target_sizes[y, x, 0] < max_size:
            if self.target_sizes[y, x, 0] + max_size // 2 < output_size // 2:
              self.target_sizes[y, x, 2] = self.target_sizes[y, x, 0] + max_size
            else:
              self.target_sizes[y, x, 0] = self.target_sizes[y, x, 2] - max_size
          if self.target_sizes[y, x, 3] - self.target_sizes[y, x, 1] < max_size:
            if self.target_sizes[y, x, 1] + max_size // 2 < output_size // 2:
              self.target_sizes[y, x, 3] = self.target_sizes[y, x, 1] + max_size
            else:
              self.target_sizes[y, x, 1] = self.target_sizes[y, x, 3] - max_size

          if self.target_sizes[y, x, 2] - self.target_sizes[y, x, 0] != max_size \
             or self.target_sizes[y, x, 3] - self.target_sizes[y, x, 1] != max_size:
            print(self.target_sizes[y, x])

      with tf.name_scope('training'):
        loss = tf.reduce_mean((target_batch - preds) ** 2)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

    pf_area = self.config['projective_field_size'] ** 2
    
    if mode == self.MODE.VALIDATE:
      with tf.name_scope('stats'):
        # Mean absolute error
        pred_counts = tf.reduce_sum(preds / pf_area, axis=[1, 2])
        target_counts = tf.reduce_sum(target_batch / pf_area, axis=[1, 2])
        mae = tf.reduce_sum(tf.abs(pred_counts - target_counts))
        mae_avg = utilities.RunningAverage('mae', mae,
                                       summary_args={'collections': ['stats_summaries']})

        # Accuracy
        acc = tf.reduce_mean(tf.cast(tf.abs(tf.reduce_mean(preds - target_batch, [1, 2])),
                                          tf.float32))
        acc_avg = utilities.RunningAverage('accuracy', acc,
                                       summary_args={'collections': ['stats_summaries']})

      with tf.control_dependencies([mae_avg.update_op, acc_avg.update_op]):
        self.valid_op = tf.no_op()  # All we need is the control dependencies above.

      self.stats_reset_op = tf.group(mae_avg.reset_op, acc_avg.reset_op)
        
      self.score = mae_avg.value

      self.debug_preds = preds
      self.debug_inputs = image_batch
      self.debug_targets = target_batch

    if mode == self.MODE.TEST:
      pf_area = (self.config['projective_field_size'] / self.config['stride']) ** 2
      pred_counts = tf.reduce_sum(preds / pf_area, axis=[1, 2])
      
      self.test_op = (pred_counts, tid_batch)
      

  def build_train_input_graph(self, training=False):
    '''Create input tfrecord tensors.
    Args:
        graph : Current graph
    Returns:
      
    Raises:
      RuntimeError: if no files found.
    '''

    filenames = tf.gfile.Glob(os.path.join(self.config['data_dir'], '*.tfrecords'))
    if not filenames:
      raise RuntimeError('No .tfrecords files found.')

    validation_index = round(self.config['train_val_split'] * len(filenames))
    if training:
      filenames = filenames[validation_index:]
    else:
      filenames = filenames[:validation_index]

    if self.config['count_examples']:
      print('Counting records...')
      examples_nb = 0
      for filename in filenames:
        for record in tf.python_io.tf_record_iterator(filename):
           examples_nb += 1
      print('  There are %i records in the %s set.' % (examples_nb,
                         'train' if training else 'validation'))
      self.config['train_examples_nb' if training else 'valid_examples_nb'] = examples_nb
      
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature_dict = {
      'image/width':    tf.FixedLenFeature([], tf.int64),
      'image/height':   tf.FixedLenFeature([], tf.int64),
      'image/scale':    tf.FixedLenFeature([], tf.float32),
      'image':          tf.FixedLenFeature([1], tf.string),
      'coords/length':  tf.FixedLenFeature([], tf.int64),
      'coords':         tf.VarLenFeature(tf.int64)           
    }

    features = tf.parse_single_example(serialized_example, features=feature_dict)

    width = tf.cast(features['image/width'], tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    channels = tf.constant(self.config['channels'], tf.int32)
    image = features['image']
    len_coords = tf.cast(features['coords/length'], tf.int32)
    coords = features['coords']
    
    # Coordinates
    coords_sparse = tf.sparse_tensor_to_dense(coords, default_value=0)    
    coords = tf.reshape(coords_sparse, [len_coords, 3])  # cls, row, col

    image_buffer = tf.reshape(features['image'], shape=[])    
    image = tf.image.decode_png(image_buffer, channels=self.config['channels'])
    image = tf.cast(image, tf.float32) / 255.
    
    angle = tf.random_uniform([], -.07, .07)
    shear_x = tf.random_uniform([], -.07, .07)
    shear_y = tf.random_uniform([], -.07, .07)
    scale = tf.random_uniform([], 0.9, 1.00)

    image, target = tf.py_func(self.preprocessExample, 
                               [image, coords, angle, shear_x, shear_y, scale],
                               [tf.float32, tf.float32], stateful=False)

    # Static shapes are required for the network.
    image_size = self.config['tile_size'] + 2 * self.config['contextual_pad']
    image.set_shape([image_size, image_size, self.config['channels']])
    s = self.config['projective_field_size']
    unpadded_size = self.config['output_size']
    target_size = 3 + unpadded_size + 2 * s
    target.set_shape([target_size, target_size, self.config['cls_nb']])

    # Normalize mean and variance or bring into the [-1, 1] range
    if self.config['normalize_inputs']:
      image = (image - self.data_mean) / self.data_var
    else:
      image = image * 2. - 1.

    if training:
      image_batch, target_batch = tf.train.shuffle_batch(
        [image, target],
        self.config['batch_size'],
        min_after_dequeue=self.config['min_after_dequeue'],
        num_threads=self.config['train_threads_nb'],
        capacity=512 * self.config['batch_size'])
    else:
      # No need for shuffling in case of data validation.
      image_batch, target_batch = tf.train.batch(
        [image, target],
        self.config['batch_size'],
        num_threads=self.config['train_threads_nb'],
        capacity=512 * self.config['batch_size'])

    return image_batch, target_batch, tf.constant(0)
  

  def inc_region(self, dst, y_min, x_min, y_max, x_max):
    '''Incremets dst in the specified region. Runs fastest on np.int8, but not much slower on
    np.int16.'''

    if y_max - y_min <= 0 or x_max - x_min <= 0:
      return

    dst[y_min:y_max, x_min:x_max] += 1

  
  def generateCountMaps(self, coords):
    '''Generates a count map for the provided list of coordinates.
    '''

    s = self.config['projective_field_size']
    unpadded_size = self.config['output_size']
    target_size = 3 + unpadded_size + 2 * s 
    countMaps = np.zeros((self.config['cls_nb'], target_size, target_size), dtype=np.int16)

    for coord in coords:
      y = coord[1] - self.config['contextual_pad']
      x = coord[2] - self.config['contextual_pad']
      if y >= 0 and y < self.config['tile_size'] and \
         x >= 0 and x < self.config['tile_size']:
        
        self.inc_region(countMaps[coord[0]], *self.target_sizes[y, x])

    return np.moveaxis(countMaps, 0, -1).astype(np.float32)
