import sys
import os, copy
import tensorflow as tf
import numpy as np

import models.utilities as utilities

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'tf_models', 'slim'))
sys.path.append(os.path.join(script_dir, '..', 'data'))

import data_preparation

import tensorflow.contrib.slim as slim
from nets import inception_v4

from models import BasicModel

class ContextualInceptionModel(BasicModel):

  def get_default_config(self):
    config = BasicModel.get_default_config(self)
    model_config = {
      'stride': 1,
      'inception_v4_checkpoint_file': os.path.join(script_dir, '..',
                                                   'data', 'inception_v4.ckpt'),
      'batch_norm_decay': 0.9,
      'batch_norm_epsilon': 0.001,
      'output_size': 29,  # Projective field size at Mixed_5b
      'pad': 32,
      'receptive_field_size': 2 * 33,
      'projective_field_size': 7,
      'target_context_pad': 23,  # ~(7./2.) / (128./33.)
      
      'target_embedding_pad': 16,
      # The contextual pad should be less than 48, which is half of the
      # difference of the receptive fields of Mixed_5a and Mixed_5d, i.e. the
      # number of pixels that the additional receptive field can look ahead. It
      # should also not include so much black borders from data augmentation.
      'contextual_pad': 32,
      # Excluding the additional pad needed in order to get the full receptive
      # field into view.
      'large_contextual_pad_unpadded': 128,
      'large_contextual_pad': 128 + 66,
      'normalize_inputs': False,
      'batch_size': 32,
      'hidden_state_size': 96,
      'draw_border': True,
      'loss_function': 'l2',
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
    if mode != self.MODE.TEST:
      image_batch, target_batch, large_target_batch = inputs
    else:
      image_batch, tid_batch = inputs

    with tf.name_scope('model') as scope:
      pad = self.config['receptive_field_size'] - self.config['contextual_pad']
      image_batch = tf.pad(image_batch, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

      def define_loss(preds, target):
        loss = None
        if self.config['loss_function'] == 'l1':
          loss = tf.reduce_mean(tf.abs(target - preds) ** 1.7)
        elif self.config['loss_function'] == 'l2':
          loss = tf.reduce_mean((target - preds) ** 2)
        else:
          raise ValueError('Loss function %s is not supported.' % self.config['loss_function'])

        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        return loss

      batch_norm_params = {
        'decay': self.config['batch_norm_decay'],
        'epsilon': self.config['batch_norm_epsilon'],
        'is_training': (mode == self.MODE.TRAIN),
      }
      with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                          normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=(mode == self.MODE.TRAIN)):

          # Instantiate Inception-v4 up to and including Mixed_5a.
          net, end_points = inception_v4.inception_v4_base(image_batch,
                                                           final_endpoint='Mixed_5a')

          # 3 x Inception-A blocks, corresponding to Mixed_5b, Mixed_5c, Mixed_5d.
          blocks = {}
          for idx in range(3):
            block_scope = 'Mixed_5' + chr(ord('b') + idx)
            net = inception_v4.block_inception_a(net, block_scope)

          net = slim.conv2d(net, 96, [1, 1], scope='Conv2d1_1x1', padding='SAME')
          net = slim.conv2d(net, 96, [3, 3], scope='Conv2d2_3x3', padding='SAME')

      preds = slim.conv2d(net, self.config['cls_nb'], [3, 3], scope='Conv2d3_3x3_preds',
                          padding='SAME', activation_fn=None, normalizer_fn=None)
        
      if mode == self.MODE.TRAIN:
        define_loss(preds, target_batch)

      # Debugging
      self.debug_sequence = []
      if mode != self.MODE.TRAIN:
        self.debug_sequence.append(preds)

      def recurrence(features, hidden, preds):
        preds = tf.stop_gradient(preds)
        
        with tf.name_scope(scope, 'Recurrence', [features, hidden, preds]):
          embed_pad = self.config['target_embedding_pad']
          padded_preds = tf.pad(preds, [[0, 0], [embed_pad, embed_pad],
                                        [embed_pad, embed_pad], [0, 0]])
          if mode == self.MODE.TRAIN:
            padded_preds = large_target_batch + padded_preds

          net = padded_preds
          with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                              normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=(mode == self.MODE.TRAIN)):
              
              # Average pooling to blur the squres to make ground truth look
              # more like the predictions.
              # net = slim.avg_pool2d(net, 16, [3, 3], 1, scope='AvgPool', padding='VALID')
              net = slim.conv2d(net, 16, [3, 3], rate=1, scope='Dilated0', padding='VALID')
              net = slim.conv2d(net, 16, [3, 3], rate=2, scope='Dilated1', padding='VALID')
              net = slim.conv2d(net, 16, [3, 3], rate=3, scope='Dilated2', padding='VALID')
              net = slim.conv2d(net, 16, [3, 3], rate=4, scope='Dilated3', padding='VALID')
              net = slim.conv2d(net, 32, [3, 3], rate=5, scope='Dilated4', padding='VALID')
              net = slim.conv2d(net, 32, [3, 3], rate=1, scope='Dilated5', padding='VALID')

              # 32 + 384 + 384 + 6
              net = tf.concat([net, hidden, features, preds], axis=3)

              # 3 x Inception-A blocks, corresponding to Mixed_5b, Mixed_5c, Mixed_5d.
              for idx in range(3):
                block_scope = 'Mixed_5' + chr(ord('b') + idx)
                net = inception_v4.block_inception_a(net, block_scope)

              hidden = slim.conv2d(net, self.config['hidden_state_size'],
                                   [1, 1], scope='Conv2d0_1x1', padding='SAME')

              net = slim.conv2d(net, 96, [1, 1], scope='Conv2d1_1x1', padding='SAME')
              net = slim.conv2d(net, 96, [3, 3], scope='Conv2d2_3x3', padding='SAME')

          preds = slim.conv2d(net, self.config['cls_nb'], [3, 3], scope='Conv2d3_3x3_preds',
                              padding='SAME', activation_fn=None, normalizer_fn=None)

          if mode != self.MODE.TRAIN:
            self.debug_sequence.append(preds)

          loss = None
          if mode == self.MODE.TRAIN:
            loss = define_loss(preds, target_batch)

          return preds, loss, hidden

      # 3 x Recurrent context blocks
      rnn_template = tf.make_template('rnn_shared_variables', recurrence)
      
      hidden_shape = [dim.value for dim in end_points['Mixed_5a'].get_shape()]
      hidden_shape[-1] = self.config['hidden_state_size']
      hidden = tf.zeros(hidden_shape)

      final_loss = None
      for idx in range(3):
        preds, final_loss, hidden = rnn_template(end_points['Mixed_5a'], hidden, preds)

    if mode == self.MODE.TRAIN:
      with tf.name_scope('stats'):
        tf.summary.scalar('final_loss', final_loss, collections=['train_summaries'])

        pf_area = self.config['projective_field_size'] ** 2
        target_counts = tf.reduce_sum(target_batch / pf_area, axis=[1, 2])
        pred_counts = tf.reduce_sum(preds / pf_area, axis=[1, 2])
        mae = tf.reduce_sum(tf.abs(pred_counts - target_counts))
        tf.summary.scalar('mae', mae, collections=['train_summaries'])
      
      def build_target_cache(input_shape, size, offset, equalize=True):
        max_size = 0
        output_size = 0
        sizes = np.empty((size, size, 4), dtype=np.int32)
        for y in range(size):
          input_r = utilities.Rect(y + offset, 0,
                                   y + offset, 0,
                                   input_shape[1], input_shape[2])
          
          r = utilities.calc_projective_field(image_batch.name, end_points['Mixed_5a'].name,
                                              input_r)
          
          sizes[y, :, 0] = r.min_y
          sizes[:, y, 1] = r.min_y
          sizes[y, :, 2] = r.max_y + 1  # For Python ranges and indices
          sizes[:, y, 3] = r.max_y + 1  # which exclude the upper bound.
          
          output_size = r.h
          
          if r.height > max_size:
            max_size = r.height

        if self.config['debug']:
          print('Projective field size: %i' % max_size)
          print('Output size: %i' % output_size)

        if equalize:
          # Make all projective fields the same size.
          for y in range(size):
            for x in range(size):
              if sizes[y, x, 2] - sizes[y, x, 0] < max_size:
                if sizes[y, x, 0] + max_size // 2 < output_size // 2:
                  sizes[y, x, 2] = sizes[y, x, 0] + max_size
                else:
                  sizes[y, x, 0] = sizes[y, x, 2] - max_size
              if sizes[y, x, 3] - sizes[y, x, 1] < max_size:
                if sizes[y, x, 1] + max_size // 2 < output_size // 2:
                  sizes[y, x, 3] = sizes[y, x, 1] + max_size
                else:
                  sizes[y, x, 1] = sizes[y, x, 3] - max_size

              if sizes[y, x, 2] - sizes[y, x, 0] != max_size \
                 or sizes[y, x, 3] - sizes[y, x, 1] != max_size:
                print(sizes[y, x])

        return sizes, max_size

      self.target_sizes, self.config['projective_field_size'] = build_target_cache(
        [dim.value for dim in image_batch.get_shape()],
        size=self.config['tile_size'],
        offset=self.config['receptive_field_size'],
      )
      
      large_input_size = self.config['tile_size'] + 2 * self.config['large_contextual_pad']
      self.target_sizes_large, _ = build_target_cache(
        [1, large_input_size, large_input_size, 1],
        size=self.config['tile_size'] + 2 * self.config['large_contextual_pad_unpadded'],
        offset=self.config['receptive_field_size'],
        equalize=False,  # The area content of these squares does not matter.
      )
    
    if mode == self.MODE.VALIDATE:
      with tf.name_scope('stats'):
        # Mean absolute error
        pf_area = self.config['projective_field_size'] ** 2
        target_counts = tf.reduce_sum(target_batch / pf_area, axis=[1, 2])
        pred_counts = tf.reduce_sum(preds / pf_area, axis=[1, 2])
        mae = tf.reduce_sum(tf.abs(pred_counts - target_counts))
        mae_avg = utilities.RunningAverage('mae', mae,
                                       summary_args={'collections': ['stats_summaries']})

        # Accuracy
        acc = tf.reduce_mean(tf.cast(tf.abs(tf.reduce_mean(preds - target_batch, [1, 2])),
                                          tf.float32))
        acc_avg = utilities.RunningAverage('accuracy', acc,
                                       summary_args={'collections': ['stats_summaries']})

      self.valid_op = tf.group(mae_avg.update_op, acc_avg.update_op)
      self.stats_reset_op = tf.group(mae_avg.reset_op, acc_avg.reset_op)
      self.score = mae_avg.value

      # Debugging
      self.debug_preds = preds
      self.debug_targets = target_batch
      self.debug_large_targets = large_target_batch

      embed_pad = self.config['target_embedding_pad']
      reduced_large_target_batch = tf.reduce_sum(large_target_batch, axis=3, keep_dims=True)
      reduced_target_batch = tf.reduce_sum(tf.pad(target_batch, [[0, 0],
                                                                 [embed_pad, embed_pad],
                                                                 [embed_pad, embed_pad],
                                                                 [0, 0]]),
                                           axis=3, keep_dims=True)
      self.debug_combined_preds = tf.concat([reduced_large_target_batch,
                                             reduced_target_batch,
                                             tf.zeros_like(reduced_target_batch)], axis=3)
      
    if not mode == self.MODE.TRAIN:
      # Debugging
      self.debug_inputs = image_batch

    if mode == self.MODE.TEST:
      pf_area = self.config['projective_field_size'] ** 2
      cond = tf.greater(preds, tf.fill(tf.shape(preds), 0.05))
      preds = tf.where(cond, preds, tf.zeros(tf.shape(preds)))
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
    scale = tf.random_uniform([], 1. / 1.05, 1.05)

    image, target, large_target = tf.py_func(self.preprocessExample, 
                               [image, coords, angle, shear_x, shear_y, scale],
                               [tf.float32] * 3, stateful=False)

    # Static shapes are required for the network.
    image_size = self.config['tile_size'] + 2 * self.config['contextual_pad']
    image.set_shape([image_size, image_size, self.config['channels']])
    s = self.config['projective_field_size']
    unpadded_target_size = self.config['output_size']
    target_size = 3 + unpadded_target_size + 2 * s
    target.set_shape([target_size, target_size, self.config['cls_nb']])
    target_size = 78
    large_target.set_shape([target_size, target_size, self.config['cls_nb']])

    # Normalize mean and variance or bring into the [-1, 1] range
    if self.config['normalize_inputs']:
      image = (image - self.data_mean) / self.data_var
    else:
      image = image * 2. - 1.

    if training:
      image_batch, target_batch, large_target_batch = tf.train.shuffle_batch(
        [image, target, large_target],
        self.config['batch_size'],
        min_after_dequeue=self.config['min_after_dequeue'],
        num_threads=self.config['train_threads_nb'],
        capacity=512 * self.config['batch_size'])
    else:
      # No need for shuffling in case of data validation.
      image_batch, target_batch, large_target_batch = tf.train.batch(
        [image, target, large_target],
        self.config['batch_size'],
        num_threads=self.config['train_threads_nb'],
        capacity=512 * self.config['batch_size'])

    return image_batch, target_batch, large_target_batch
  

  def inc_region(self, dst, y_min, x_min, y_max, x_max):
    '''Incremets dst in the specified region. Runs fastest on np.int8, but not much slower on
    np.int16.'''

    if y_max - y_min <= 0 or x_max - x_min <= 0:
      return

    dst[y_min:y_max, x_min:x_max] += 1

  
  def preprocessExample(self, image, coords, angle, shear_x, shear_y, scale):
    size_in = image.shape[0]
    size_out = self.config['tile_size'] + 2 * self.config['contextual_pad']
    
    # h = base64.b64encode(struct.pack(">q", hash(image.tostring()))).decode()

    # data_preparation.imshow(image, coords=coords, save=True, title='%s_preprocessExampleA' %h)
    
    image = self.applyLinearTransformToImage(image, angle, shear_x, shear_y, scale, size_out)
    image = self.applyColorAugmentation(image, self.config['aug_color_std'], \
                                        self.config['aug_gamma_factor'])
    coords[:, 1:] = self.applyLinearTransformToCoords(coords[:, 1:], angle, shear_x,
                                                      shear_y, scale, size_in, size_out)
    target = self.generateCountMaps(coords)
    large_target = self.generateLargeCountMaps(coords)

    if self.config['draw_border'] and self.config['contextual_pad'] > 0:
      image = self.draw_border(image, self.config['contextual_pad'], self.config['tile_size'])
      
    # data_preparation.imshow(image, coords=coords, save=True, title='%s_preprocessExampleB' % h)
    # t = np.concatenate(np.moveaxis(target, -1, 0))
    # data_preparation.imshow(t, normalize=True, save=True, title='%s_preprocessExampleC' % h)
    
    return image.astype(np.float32), target, large_target


  def generateLargeCountMaps(self, coords):
    '''Generates a count map for the provided list of coordinates.
    '''
    c = self.config['target_context_pad']
    target_size = 3 + self.config['output_size'] + 2 * c
    count_maps = np.zeros((self.config['cls_nb'], target_size, target_size), dtype=np.int16)

    # We want coordinates relative to the fully padded large size. For that we
    # first get coordinates wrt the unpadded tile and then set the upper left
    # corner of the large size as the origin.
    pad = self.config['large_contextual_pad']
    shift = - self.config['contextual_pad'] + pad
    r = self.config['receptive_field_size']
    tile_size = self.config['tile_size']
    size = tile_size + 2 * pad
    for coord in coords:
      y = coord[1] + shift
      x = coord[2] + shift
      if (not (y >= pad and y < pad + tile_size and \
               x >= pad and x < pad + tile_size)) and \
         y >= r and y < size - r and \
         x >= r and x < size - r:
        
        self.inc_region(count_maps[coord[0]], *self.target_sizes_large[y - r, x - r])

    # t = np.concatenate(count_maps)
    # data_preparation.imshow(t, normalize=True, save=True, title='large')
    
    return np.moveaxis(count_maps, 0, -1).astype(np.float32)

  
  def generateCountMaps(self, coords):
    '''Generates a count map for the provided list of coordinates.
    '''
    s = self.config['projective_field_size']
    target_size = 3 + self.config['output_size'] + 2 * s 
    count_maps = np.zeros((self.config['cls_nb'], target_size, target_size), dtype=np.int16)

    shift = - self.config['contextual_pad']
    size = self.config['tile_size']
    for coord in coords:
      y = coord[1] + shift
      x = coord[2] + shift
      if y >= 0 and y < size and \
         x >= 0 and x < size:
        
        self.inc_region(count_maps[coord[0]], *self.target_sizes[y, x])

    return np.moveaxis(count_maps, 0, -1).astype(np.float32)
    
