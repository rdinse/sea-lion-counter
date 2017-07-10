import os, copy, ast, sys
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import locale
import time
import re
import subprocess
from collections import namedtuple
import struct, base64

# from google.protobuf.json_format import MessageToJson
# from tensorflow.core.framework import op_def_pb2
# from google.protobuf import text_format

from tensorflow.python.client import device_lib

from scipy.ndimage import interpolation as ndii
import skimage
import cv2
import math

script_dir = os.path.dirname(os.path.realpath(__file__))
configs_dir = os.path.join(script_dir, 'configs')
root_dir = os.path.join(script_dir, '..')

sys.path.append(root_dir)

sys.path.append(os.path.join(script_dir , '../cocob/optimizer'))
from cocob.optimizer import COCOB

sys.path.append(os.path.join(script_dir, '../data'))
import data_preparation

import models.utilities as utilities


class BasicModel(object):

  MODE = namedtuple('MODE', ['TRAIN', 'VALIDATE', 'TEST'])(0, 1, 2)

  def get_default_config(self):
    '''This function returns a dictionary of default settings for the model.
    It is meant to be overridden in the following way:
    
      config = BasicModel.get_default_config(self)
      model_config = { ... }
      config.update(model_config)
      return config

    Returns:
      A dictionary of the model configurations.
    
    '''
    return {
      # Whether to load the best configuration at start-up
      'load_best_config': False,
      
      # Whether to run the program in debugging mode.
      'debug': True,
      # How often to run the debugging code in the training loop.
      'debug_output_every': 100,
      # Whether to clear existing debug info.
      'clear_debug': True,
      # Whether to clear existing results.
      'clear_results': False,
      # Run profiling metadata.
      'metadata_every': 3000,
      # How often to query the summaries.
      'train_log_every': 10,
      # Whether to report histograms of the gradients.
      'gradient_summaries': False,
      # Whether to log the placement on the devices in the console.
      'log_device_placement': False,
      # Whether to clear existing results.
      'clear_results': False,
      
      # Name of the checkpoint directory that the latest checkpoint is chosen
      # from (give that no checkpoint_file is provided).
      'checkpoint_dir': None,
      # Name of the checkpoint file at the given path that should be loaded.
      'checkpoint_file': None,
      # Whether to load the variables from a checkpoint file rather than newly
      # initilizing them.
      'load_checkpoint': False,
      # Global step at which the model was saved last.
      'last_checkpoint': -1,
      # Number of checkpoints to keep.
      'max_to_keep': 999,
     
      # Whether to recompute the example numbers at start-up.
      'count_examples': False,
      # Number of examples in the training set.
      'train_examples_nb': 134351,
      # Number of examples in the validation set.
      'valid_examples_nb': 14936,
      
      # Whether to save the model upon a keyboard interrupt.
      'save_after_interrupt': True,
      # How often to save.
      'save_every': 2000,
      # How often to validate. 
      'validate_every': 5000,
      # How many steps to run the training loop for.
      'max_iter': 0,
      
      # Size of the test tiles (should be large to avoid double-counting at the
      # seams).
      'test_tile_size': 512,
      # Batch size used for testing.
      'test_batch_size': 16,
      # Number of test-time augmentations (4 times rot90).
      'test_augmentation_nb': 1,
      # Number of threads used during testing.
      'test_threads_nb': 4,
      # How often to report progress statistics during testing.
      'test_report_every': 1,

      # Factor by which the training data is scaled by.
      'scale_factor': data_preparation.scale_factor,
      # Size of the training tiles.
      'tile_size': data_preparation.tile_size,
      # Margin of the training tiles.
      'tile_margin': data_preparation.tile_margin,
      # Additional margin for context.
      'contextual_margin': 0,
      # Number of classes in the training data.
      'cls_nb': data_preparation.cls_nb,
      # Size of the receptive fiels (TODO).
      'receptive_field_size': 32,
      # Number of channels of the input images.
      'channels': 3,
      # Number of GPUs to use.
      'num_gpus': 1,
      # Batch size used for training.
      'batch_size': 8,
      # Number of threads to run the tf.train.shuffle_batch queue with.
      'train_threads_nb': 8,
      # Minimal examples to shuffle at a time in the tf.train.shuffle_batch
      # queue.
      'min_after_dequeue': 256,
      # Whether to normalize the inputs with data mean and variance.
      'normalize_inputs': False,

      # Directory that contains the preprocessed training data.
      'data_dir': os.path.join(root_dir, 'data'),
      # Directory that contains the training and test data.
      'input_dir': os.path.join(root_dir, 'input'),
      # Directory that contains all results of the current run.
      'results_dir': os.path.join(root_dir, 'results', self.__class__.__name__
                                  + time.strftime('_%Y-%m-%d_%H-%M-%S', time.gmtime())),
      # Directory that debugging information is written to.
      'debug_dir': os.path.join(root_dir, 'debug'),
      # Directory that the test data (JPEGs) is read from.
      'test_dir': 'Train',  # Can be Test or Train (for testing the test run).
      # Name of the ID column in the predictions data frame/CSV file.
      'test_id_col_name': 'train_id',

      # Ratio by which to split the training data into train/validation sets.
      'train_val_split': 0.1,

      # File containing mean and covariance information for data augmentation
      # and input normalization.
      'pca_file': os.path.join(root_dir, 'data', 'pca.npz'),

      # Random seed for reproducible results.
      'random_seed': 0,

      # Whether to use early stopping.
      'early_stopping': False,
      # The patience is often set somewhere between 10 and 100 (10 or 20 is
      # more common).
      'early_stopping_patience': 15,
      # Amount by which the score needs to improve to count as a patience step.
      'early_stopping_margin': 0.,
      # Whether the score needs to be minimized 'min' or maximized 'max'.
      'early_stopping_mode': 'min',

      # Amount by which the eigenvalues are scaled during color data
      # augmentation.
      'aug_color_std': 0.55,
      # Amount by which the sum of the eigen* perturbation is divided by for
      # gamma adjustment during data augmentation.
      'aug_gamma_factor': 3,

      'draw_border': True,
    }
  

  def __init__(self, config={}):
    '''Initializes a new basic model.
    '''
    print('Constructing model...')

    # Prepare configurations.
    config_ = self.get_default_config()
    
    if config['load_best_config']:
      config_.update(self.get_best_config())

    config_.update(config)

    self.config = copy.deepcopy(config_)
    del config, config_

    if self.config['debug']:
      print('Configuration:')
      print(json.dumps(self.config, sort_keys=True, indent=4, separators=(',', ': ')))

    # Preparing workspace.
    if self.config['clear_results'] and tf.gfile.Exists(self.config['results_dir']):
      tf.gfile.DeleteRecursively(self.config['results_dir'])
    tf.gfile.MakeDirs(self.config['results_dir'])
    if self.config['clear_debug'] and tf.gfile.Exists(self.config['debug_dir']):
      print('Clearing debug directory.')
      tf.gfile.DeleteRecursively(self.config['debug_dir'])
    tf.gfile.MakeDirs(self.config['debug_dir'])

    # Preparing random seed.
    tf.set_random_seed(self.config['random_seed'])
    np.random.seed(self.config['random_seed'])

    # Load eigen decomposition, mean and covariance value for
    # data normalization and augmentation.
    pca = np.load(self.config['pca_file'])
    self.data_evecs = pca['evecs']
    self.data_evals = pca['evals']
    self.data_mean = pca['mean']
    self.data_cov = pca['cov']
    self.data_var = np.diag(self.data_cov)

    # Initialize early stopping.
    self.best_global_step = -1
    self.best_score = np.inf if self.config['early_stopping_mode'] == 'min' else -np.inf

    # Check GPU devices.
    device_list = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']

    if self.config['debug']:
      print('Available GPU devices:' + ('' if len(device_list) else ' None'))
      for device in device_list:
        print('  ' + device)
    if len(device_list) == 0:
      print('Warning: No GPU devices detected. '
            + 'The model will be placed on CPUs instead (soft placement).')
    elif len(device_list) < self.config['num_gpus']:
      raise ValueError('Error: %i GPUs have been requested, but only %i are available.'
            % (len(device_list), range(self.config['num_gpus'])))

    # Build coordination.
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options,
                                 # operation_timeout_in_ms=5000,  # For debug only.
                                 allow_soft_placement=True,
                                 log_device_placement=self.config['log_device_placement']
                                                      and self.config['debug'])
    self.graph = tf.Graph()
    self.sess = tf.Session(config=sess_config, graph=self.graph)

    # Build graph.
    with self.graph.as_default():
      global_step = tf.Variable(0, trainable=False, name='global_step',
                                collections=[tf.GraphKeys.GLOBAL_STEP,
                                             tf.GraphKeys.GLOBAL_VARIABLES])
      with tf.name_scope('inputs'):
        if self.config['task'] == 'train':
          with tf.name_scope('train'):
            inputs_train = self.build_train_input_graph(training=True)
          with tf.name_scope('valid'):
            inputs_valid = self.build_train_input_graph(training=False)
        else:
          with tf.name_scope('test'):
            inputs_test = self.build_test_input_graph()

      model_template = tf.make_template('shared_variables', self.build_model_graph)
      if self.config['task'] == 'train':
        with tf.name_scope('training_instance'):
          model_template(inputs=inputs_train, mode=self.MODE.TRAIN)
        with tf.name_scope('validation_instance'):
          model_template(inputs=inputs_valid, mode=self.MODE.VALIDATE)
      else:
        with tf.name_scope('testing_instance'):
          model_template(inputs=inputs_test, mode=self.MODE.TEST)

      already_initialized = self.init_variables()

      if self.config['task'] == 'train':
        with tf.name_scope('optimization'):
          # We collect the gradients separately in case we are doing multi-GPU
          # training and need to accumulate the gradients over multiple model
          # replica.
          
          self.opt = COCOB()
          # self.opt = tf.train.RMSPropOptimizer(learning_rate=0.001)
          combined_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
          grads = self.opt.compute_gradients(combined_loss)

          tf.summary.scalar('loss', combined_loss, collections=['train_summaries'])

          self.check_numerics = tf.check_numerics(combined_loss,
                                                  'Model diverged with loss = NaN')

          with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = self.opt.apply_gradients(grads, global_step=global_step)

      if self.config['task'] == 'train' and self.config['debug']:
        with tf.name_scope('model/train'):
          for grad, var in grads:
            if grad is not None:
              tf.summary.histogram(var.op.name + '/gradients', grad,
                                   collections=['train_summaries'])
        
      self.sw = tf.summary.FileWriter(self.config['results_dir'], graph=self.graph)
      if self.config['task'] == 'train':
        self.train_summaries = tf.summary.merge(tf.get_collection('train_summaries'))
        self.stats_summaries = tf.summary.merge(tf.get_collection('stats_summaries'))

        # TODO QueueSize summary
        # https://stackoverflow.com/questions/40191367/tensorflow-get-amount-of-samples-in-queue

      self.saver = tf.train.Saver(var_list=tf.global_variables(),
                                  max_to_keep=self.config['max_to_keep'])

      to_be_initialized = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      to_be_initialized = [var for var in to_be_initialized if var not in already_initialized]
      self.sess.run(tf.group(tf.variables_initializer(to_be_initialized),
                             tf.local_variables_initializer()))
      
      # Load checkpoints.
      if self.config['load_checkpoint']:
        if self.config['checkpoint_file'] is not None:
          checkpoint = self.config['checkpoint_file']
        else:
          checkpoint = tf.train.latest_checkpoint(self.config['checkpoint_dir'])
          if checkpoint is not None:
            checkpoint = checkpoint.model_checkpoint_path

        if checkpoint is not None:
          self.saver.restore(self.sess, checkpoint)
          print('Restored the model from: %s' % checkpoint)
        else:
          raise FileNotFoundError('No checkpoint found.')
      else:
        print('No checkpoint found. Initializing a new model.')
        self.init()

      # Print number of trainable parameters.
      variable_params = self.sess.run([tf.reduce_prod(tf.shape(v))
                                       for v in tf.trainable_variables()])
      if self.config['debug']:
        print('Number of parameters of all trainable variables:')
        for i, v in enumerate(tf.trainable_variables()):
          print('  {}: {:,}'.format(v.name, variable_params[i]))
      print('Total number of parameters: {:,}'.format(np.sum(variable_params)))

      # Finalize the graph.
      self.graph.finalize()

      # Starts the queues, including loading and caching of the training data.
      self.coord = tf.train.Coordinator()
      self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    # Also finalize the default graph.
    tf.get_default_graph().finalize()
    
    print('The Model has been constructed.')


  def get_best_config(self):
    config_name = 'best_' + self.__class__.__name__ + '_config.json'
    with open(os.path.join(configs_dir, config_name), 'r') as f:
      return json.loads(f.read())


  def save_best_config(self, config=None):
    config_name = 'best_' + self.__class__.__name__ + '_config.json'
    with open(os.path.join(configs_dir, config_name), 'w') as f:
      json.dump(self.config, f, cls=utilities.NumPyCompatibleJSONEncoder)
    

  @staticmethod
  def get_random_hps(fixed_params={}):
    '''This function returns a random sample of parameters for hyper-parameter
    search.  It is a static function such that the random parameters can be
    sampled before the model is instantiated.
    '''
    raise Exception('The get_random_hps function must be overriden')

  
  def build_model_graph(self, inputs, mode, reuse=False):
    '''This function is supposed to construct the graph of the model.  It
    should add all losses to the 'losses' collection.
    '''
    raise Exception('The build_model_graph function must be overriden')


  def build_train_input_graph(self, training=False):
    '''Create input tfrecord tensors.
    Args:
        graph : Current graph
    Returns:
      
    Raises:
      RuntimeError: if no files found.
    '''

    suffix = train_suffix if training else val_suffix
    filenames = tf.gfile.Glob(os.path.join(self.config['data_dir'],
                                           '*_%s.tfrecords' % suffix))
    if not filenames:
      raise RuntimeError('No .tfrecords files found.')

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
    
    angle = tf.random_uniform([], -.1, .1)
    shear_x = tf.random_uniform([], -.07, .07)
    shear_y = tf.random_uniform([], -.07, .07)
    scale = tf.random_uniform([], 1. / 1.05, 1.05)

    image, target = tf.py_func(self.preprocessExample, 
                               [image, coords, angle, shear_x, shear_y, scale],
                               [tf.float32, tf.float32], stateful=False)

    # Static shapes are required for the network.
    image_size = self.config['tile_size'] + 2 * self.config['contextual_pad']
    image.set_shape([image_size, image_size, self.config['channels']])
    target_size = 1 + image_size + 2 * (self.config['receptive_field_size'] // 2)
    target.set_shape([target_size, target_size, self.config['cls_nb']])

    # Normalize mean and variance or bring into the [-1, 1] range
    if self.config['normalize_inputs']:
      image = (image - self.data_mean) / self.data_var
    else:
      image = image * 2 - 1.

    if training:
      # Recommendation:
      # Set the capacity of the batch big enough to mix well without exhausting your RAM
      # Tens of thousands of examples usually works pretty well.
      # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
      # One example takes about 1-2 MB
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
  

  def train(self):
    '''Runs the training loop.

    Returns:
      In case max_iter is set, this function returns the final
      performance on the validation set.
    '''
    
    print('Begin training...\nUse TensorBoard to monitor the progress.')
    
    current_score = np.nan
    while not self.coord.should_stop():
      try:
        # with self.graph.as_default():
        #   print(self.sess.run(tf.get_default_graph().get_tensor_by_name("shared_variables/InceptionV4/Mixed_3a/Branch_1/Conv2d_0a_3x3/BatchNorm/beta:0") ))
        summaries = None
        
        global_step = self.sess.run(tf.train.get_global_step(self.graph))
        if self.config['debug'] and global_step > 1 and self.config['metadata_every'] > 0 \
           and global_step % self.config['metadata_every'] == 0:

          run_metadata = tf.RunMetadata()
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          summaries, _ = self.sess.run([self.train_summaries, self.train_op],
                                       options=run_options,
                                       run_metadata=run_metadata)
          self.sw.add_run_metadata(run_metadata, 'metadata_step_%d' % global_step, global_step)
          print('Global step %i: Metadata saved to summaries.' % global_step)
        elif global_step + 1 % self.config['train_log_every']:
          summaries, _ = self.sess.run([self.train_summaries, self.train_op])
        else:
          self.sess.run(self.train_op)
          
        global_step = self.sess.run(tf.train.get_global_step(self.graph))

        if summaries is not None:
          self.sw.add_summary(summaries, global_step)

        if self.config['save_every'] > 0 and global_step % self.config['save_every'] == 0:
          self.save()
          print('Global step %i: Model saved.' % global_step)

        if self.config['validate_every'] > 0 \
           and global_step % self.config['validate_every'] == 0:
          
          current_score = self.validate() 
          print('Global step %i: Validation finished with a score of %f.' %
                (global_step, current_score))
          
          if self.config['early_stopping']:
            op = np.less if mode == 'min' else np.greater
            margin = self.config['early_stopping_margin'] if mode == 'min' \
                     else -self.config['early_stopping_margin']
            if op(current_score + margin, self.best_score):
              self.best_score = current_score
              self.best_wait = 0
            else:
              if self.best_wait >= self.config['early_stopping_patience']:
                print('Global step %i: Early stopping.' % global_step)
                self.save()
                self.coord.request_stop()
              self.best_wait += 1

        if self.config['max_iter'] > 0 and self.global_step >= self.config['max_iter']:
          print('Global step %i: Maximal number of iterations reached.' % global_step)
          current_score = self.validate() 
          print('Final score: %f' % current_score)
          self.save()
          break

        # if self.config['debug'] and self.config['debug_output_every'] > 0\
        if self.config['debug_output_every'] > 0\
           and global_step % self.config['debug_output_every'] == 0:

          print('Global Step: %i' % global_step)

          try:
            s_protos = tf.Summary().FromString(summaries)
            for msg in s_protos.value:
              if 'loss' in msg.tag:
                print('  %s: %f' % (msg.tag, msg.simple_value))
          except:
            pass
          
          inpts, preds, targs, ltargs, ctargs, stargs = self.sess.run([
            self.debug_inputs,
            self.debug_preds,
            self.debug_targets,
            self.debug_large_targets,
            self.debug_combined_preds,
            self.debug_sequence,
          ])
          print('  Inputs:  max: %f, min: %f' % (np.max(inpts), np.min(inpts)))
          print('  Ouputs:  max: %f, min: %f' % (np.max(preds), np.min(preds)))
          print('  Targets: max: %f, min: %f' % (np.max(targs), np.min(targs)))
          data_preparation.imshow(inpts[0], save=True, title='debug_%ii_' % global_step,
                                  normalize=True)
          t = np.concatenate([np.concatenate(np.moveaxis(p[0], -1, 0)) for p in stargs] + \
                              [np.concatenate(np.moveaxis(targs[0], -1, 0))], axis=1)
          data_preparation.imshow(t, save=True, title='debug_%it_' % global_step, normalize=True)
          data_preparation.imshow(np.concatenate(np.moveaxis(ltargs[0], -1, 0)), save=True,
                                  title='debug_%il_' % global_step, normalize=True)
          data_preparation.imshow(ctargs[0], save=True,
                                  title='debug_%rc_' % global_step, normalize=True)
        
      except KeyboardInterrupt:
        print('Keyboard interrupt: Training stopped.')
        if self.config['save_after_interrupt']:
          self.save()
        break

    # Wait for threads to finish.
    self.coord.request_stop()
    self.coord.join(self.threads)
    self.sess.close()

    return current_score


  def validate(self):
    self.sess.run(self.stats_reset_op)

    valid_iteration = 0
    while not self.coord.should_stop() \
          and valid_iteration < self.config['valid_examples_nb'] // self.config['batch_size']:
      
      try:
        self.sess.run(self.valid_op)
        valid_iteration += 1
      except KeyboardInterrupt as e:
        print('Keyboard interrupt: Validation stopped.')
        break
    
    global_step = self.sess.run(tf.train.get_global_step(self.graph))
    self.sw.add_summary(self.sess.run(self.stats_summaries), global_step)
    return self.sess.run(self.score)


  def build_test_input_graph(self):
    '''Create input tfrecord tensors.
    
    Returns:
      Batches of images and their targets. 

    Raises:
      RuntimeError: if no files found.
    
    '''
    filenames = tf.gfile.Glob(os.path.join(self.config['input_dir'],
                                        self.config['test_dir'], '*.jpg'))

    self.test_total_nb = len(filenames) * 22 * self.config['test_augmentation_nb'] \
                         / self.config['test_batch_size']

    if not filenames:
      raise RuntimeError('No test images found.')
    
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=1)
    filename = filename_queue.dequeue()

    def filename_to_int(filename):
      return int(os.path.basename(filename)[0:-4])  # Truncate .jpg suffix.

    filenames_tids = list(map(filename_to_int, filenames))

    def read_and_split_image(filename):
      tile_size = self.config['test_tile_size']

      image = cv2.imread(filename.decode())
      scale = 1. / self.config['scale_factor']
      # Warning: (width, height)
      new_size = (int(image.shape[1] // scale), int(image.shape[0] // scale))
      image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
      image = image[..., ::-1] / 255. 

      cpad = self.config['contextual_pad']
      pad_h = tile_size * int(math.ceil(new_size[1] / float(tile_size))) - new_size[1] + 2 * cpad
      pad_w = tile_size * int(math.ceil(new_size[0] / float(tile_size))) - new_size[0] + 2 * cpad
      image = np.pad(image, [[0, pad_h], [0, pad_w], [0, 0]], 'constant')
      tiles = []
      for y in range(cpad, image.shape[0] - cpad, tile_size):
        for x in range(cpad, image.shape[1] - cpad, tile_size):
          tile = image[y - cpad:y + tile_size + cpad, x - cpad:x + tile_size + cpad]

          if self.config['draw_border'] and self.config['contextual_pad'] > 0:
            tile = self.draw_border(np.copy(tile), self.config['contextual_pad'], tile_size)
          
          # for k in range(4):
          #   tiles.append(np.rot90(tile))
          # print(tile.shape, end='')
          
          tiles.append(tile)

      all_tiles = np.stack(tiles).astype(np.float32)
      
      if self.config['normalize_inputs']:
        all_tiles = (all_tiles - self.data_mean) / self.data_var
      else:
        all_tiles = 2. * all_tiles - 1.
        
      return all_tiles

    tiles = tf.py_func(read_and_split_image, [filename], [tf.float32], stateful=False)[0]
    
    # All of the tiles for the current image need to be tagged by the same tid
    # such that the predictions can be summed up.
    tid = tf.py_func(filename_to_int, [filename], [tf.int64], stateful=False)[0]
    tids = tf.fill([tf.shape(tiles)[0], 1], tid)
    
    image_batch, tid_batch = tf.train.batch(
      [tiles, tids],
      shapes=[[self.config['test_tile_size'] + 2 * self.config['contextual_pad'],
                  self.config['test_tile_size'] + 2 * self.config['contextual_pad'],
                  self.config['channels']],
              [1]],
      enqueue_many=True,
      batch_size=self.config['test_batch_size'],
      capacity=32 * self.config['test_batch_size'],
      num_threads=self.config['test_threads_nb'])

    return image_batch, tid_batch 
  

  def test(self):
    '''Runs the testing loop.'''
    
    cls_names = list(data_preparation.sld.cls_names[:-1])  # Exclude the UNK token.
    tid_col_name = self.config['test_id_col_name']

    preds_df = pd.DataFrame(columns=[tid_col_name] + cls_names)
    
    preds = None
    tids = None
    test_iteration = 1
    while not self.coord.should_stop():
      try:
        if test_iteration % self.config['test_report_every'] == 0:
          print('Test iteration: %i of ~%i' % (test_iteration, self.test_total_nb), end='\r')

        if self.config['debug'] and self.config['metadata_every'] > 0 \
           and test_iteration % self.config['metadata_every'] == 0:
          
          run_metadata = tf.RunMetadata()
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          preds, tids = self.sess.run(self.test_op,
                                      options=run_options,
                                      run_metadata=run_metadata)
          self.sw.add_run_metadata(run_metadata, 'metadata_test_%i' % test_iteration)
          print('Metadata saved to summaries.')
        else:
          preds, tids = self.sess.run(self.test_op)

          # ((preds, tids),
          #   debug_inputs, 
          #   debug_sequence, 
          # ) = self.sess.run([
          #   self.test_op,
          #   self.debug_inputs,
          #   self.debug_sequence,
          # ])

          # data_preparation.imshow(debug_inputs[0], save=True,
          #                         title='debug_%i_' % test_iteration, normalize=True)
          # pred_maps = np.moveaxis(debug_sequence[-1][0], -1, 0)
          # data_preparation.imshow(np.concatenate(pred_maps),
          #                         save=True,
          #                         title='debug_%i_%s_%s' % (test_iteration,
          #                                                   str([(np.min(p),np.max(p)) for p in pred_maps]),
          #                                                   str(preds[0])),
          #                         normalize=True)

        # Reject negative predictions.
        # np.maximum(preds, 0., preds)

        # Account for test time augmentation.
        preds /= self.config['test_augmentation_nb']

        # Iterate over all predictions in current batch.
        for pred, tid in zip(list(preds), tids.reshape(-1)):
          query = preds_df[tid_col_name] == tid
          if query.any():
            preds_df.loc[query, cls_names] += list(pred)
          else:
            preds_df.loc[len(preds_df)] = [tid] + list(pred)
            
        test_iteration += 1
      except tf.errors.OutOfRangeError as e:
        break
      except KeyboardInterrupt as e:
        print('Keyboard interrupt: Testing stopped.')
        break

    # Wait for threads to finish.
    self.coord.request_stop()
    self.coord.join(self.threads)
    self.sess.close()

    # Convert tid to integer.
    preds_df[tid_col_name] = preds_df[tid_col_name].astype(np.int32)

    # Generate predictions file and RMSE (if targets are available).
    summary_dict = { }
    rmse = np.zeros((len(cls_names)))
    targets_file = os.path.join(self.config['input_dir'], self.config['test_dir'], 'train.csv')
    if tf.gfile.Exists(targets_file):
      targets_df = pd.read_csv(targets_file)

      rmse_counter = 0
      for pred in preds_df.itertuples():
        pred = pred[1:]  # We do not need the index.
        query = targets_df[tid_col_name] == pred[0]
        if query.any():
          rmse += (targets_df.loc[query, cls_names].values[0] - np.asarray(pred[1:])) ** 2
          rmse_counter += 1
        else:
          print('Warning: Entry with train_id = ID %i does not exist.' % pred[0])
      rmse = np.sqrt(rmse / rmse_counter)
      rmse = np.mean(rmse)  # Mean over the column-wise RMSEs
      print('RMSE: %f' % rmse)
      summary_dict['RMSE'] = rmse
    
    summary_prefix = time.strftime('%Y-%m-%d_%H-%M-%S_', time.gmtime()) 
    predictions_file = os.path.join(self.config['results_dir'],
                                 summary_prefix + 'test_predictions.csv')
    summary_file = os.path.join(self.config['results_dir'],
                                summary_prefix + 'test_summary.json')
    
    preds_df.sort_values(tid_col_name, inplace=True)
    preds_df.to_csv(predictions_file, index=False)
    
    summary_dict.update(self.config)
    with open(summary_file, 'w') as f:
      json.dump(summary_dict, f, cls=utilities.NumPyCompatibleJSONEncoder)
      
    print('Test results saved to:', os.path.join(self.config['results_dir'], summary_prefix))
    
    
  @utilities.non_interruptable
  def save(self):
    global_step = self.sess.run(tf.train.get_global_step(self.graph))

    if self.config['last_checkpoint'] == global_step:
      if self.config['debug']:
        print('Model has already been saved during the current global step.')
        return

    print('Saving to %s with global_step %d.' % (self.config['results_dir'], global_step))

    self.saver.save(self.sess, os.path.join(self.config['results_dir'], 'checkpoint'), global_step)
    self.config['last_checkpoint'] = global_step

    # Also save the configuration
    json_file = os.path.join(self.config['results_dir'], 'config.json')
    with open(json_file, 'w') as f:
      json.dump(self.config, f, cls=utilities.NumPyCompatibleJSONEncoder)


  def init_variables(self):
    '''This funciton is called only once if the model is called right after the
    graph is built, after global and local variables are initialized.  This
    function allows to re-initialize or pre-load variables e.g. for loading
    pre-trained models.  The implementation of this function is optional.
    '''
    pass


  def init(self):
    '''This funciton is called only once if the model is not loaded from a
    checkpoint.  The implementation of this function is optional.
    '''
    pass


  def applyLinearTransformToImage(self, image, angle, shear_x, shear_y, scale, size_out):
    '''Apply the image transformation specified by three parameters.

    Time it takes warping a 256 x 256 RGB image with various affine warping functions:

    * 0.25ms cv2.warpImage, nearest interpolation
    * 0.26ms cv2.warpImage, linear interpolation
    * 5.11ms ndii.affine_transform, order=0
    * 5.93ms skimage.transform._warps_cy._warp_fast, linear interpolation

    Args:
      x: 2D numpy array, a single image.
      angle: Angle by which the image is rotated.
      shear_x: Shearing factor along the x-axis by which the image is sheared.
      shear_y: Shearing factor along the x-axis by which the image is sheared.
      scale: Scaling factor by which the image is scaled.
      channel_axis: Index of axis for channels in the input tensor.

    Returns:
      A tuple of transformed version of the input and the correction scale factor.
    
    '''
    # Positions of the image center before and after the transformation in
    # pixel coordinates.
    s_out = (size_out, size_out)
    c_in = .5 * np.asarray(image.shape[:2], dtype=np.float64).reshape((2, 1))
    c_out = .5 * np.asarray(s_out, dtype=np.float64).reshape((2, 1)) 

    angle = -angle
    M_rot_inv = np.asarray([[math.cos(angle), -math.sin(angle)], \
                            [math.sin(angle),  math.cos(angle)]])
    M_shear_inv = (1. / (shear_x * shear_y - 1.)) \
                * np.asarray([[-1., shear_x], [shear_y, -1.]])

    M_inv = np.dot(M_shear_inv, M_rot_inv)  # First undo rotation, then shear.

    M_inv /= scale

    offset = c_in - np.dot(M_inv, c_out)

    # cv2.warpAffine transform according to dst(p) = src(M_inv * p + offset).
    # No need to reverse the channels because the channels are interpolated
    # separately.
    warped = cv2.warpAffine(image, np.concatenate((M_inv, offset), axis=1), s_out, 
                            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                            # flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)

    return warped


  def applyLinearTransformToCoords(self, coords, angle, shear_x, shear_y, scale, \
                                   size_in, size_out):
    '''Apply the image transformation specified by three parameters to a list of
    coordinates. The anchor point of the transofrmation is the center of the tile.

    Args:
      x: list of coordinates.
      angle: Angle by which the image is rotated.
      shear_x: Shearing factor along the x-axis by which the image is sheared.
      shear_y: Shearing factor along the x-axis by which the image is sheared.
      scale: Scaling factor by which the image is scaled.

    Returns:
      A list of transformed coordinates.
    
    '''
    s_in = (size_in, size_in)
    s_out = (size_out, size_out)
    c_in = .5 * np.asarray(s_in, dtype=np.float64).reshape((1, 2))
    c_out = .5 * np.asarray(s_out, dtype=np.float64).reshape((1, 2)) 

    M_rot = np.asarray([[math.cos(angle), -math.sin(angle)], \
                        [math.sin(angle),  math.cos(angle)]])
    M_shear = np.asarray([[1., shear_x], [shear_y, 1.]])
    M = np.dot(M_rot, M_shear)
    M *= scale  # Without translation, it does not matter whether scale is
                # applied first or last.

    coords = coords.astype(np.float64)
    coords -= c_in
    coords = np.dot(M.T, coords.T).T
    coords += c_out
    return np.round(coords).astype(np.int32)
  
  
  # tf augmentation methods
  # TODO https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/preprocessing.py


  def applyColorAugmentation(self, img, std=0.55, gamma=2.5):
    '''Applies random color augmentation following [1].  An additional gamma
    transformation is added.

    [1] Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton.  ImageNet
        Classification with Deep Convolutional Neural Networks.  NIPS 2012.
    '''

    alpha = np.clip(np.random.normal(0, std, size=3), -1.3 * std, 1.3 * std)
    perturbation = self.data_evecs.dot((alpha * np.sqrt(self.data_evals)).T)
    gamma = 1.0 - sum(perturbation) / gamma
    return np.power(np.clip(img + perturbation, 0., 1.), gamma)
    return np.clip((img + perturbation), 0., 1.)
  

  def blit_add(self, dst, src, y, x):
    ''' Adds src to the specified region in dst. '''
    
    ds = dst.shape
    ss = src.shape
    ss2 = (ss[0] // 2, ss[1] // 2)
    py = y - ss2[0]
    px = x - ss2[1] 
    dy_min = max(0, py)
    dy_max = min(ds[0], y + ss2[0])
    dx_min = max(0, px)
    dx_max = min(ds[1], x + ss2[1])
    sy_min = max(0, -py)
    sy_max = min(ss[0], ds[0] - py)
    sx_min = max(0, -px)
    sx_max = min(ss[1], ds[1] - px)

    if sy_max - sy_min <= 0 or sx_max - sx_min <= 0 \
       or dy_max - dy_min <= 0 or dx_max - dx_min <= 0:
        return

    dst[dy_min:dy_max, dx_min:dx_max] += src[sy_min:sy_max, sx_min:sx_max]


  def inc_region(self, dst, y, x, h, w):
    '''Incremets dst in the specified region. Runs fastest on np.int8, but not much slower on
    np.int16.'''

    dh, dw = dst.shape
    h2 = h // 2
    w2 = w // 2
    py = y - h2 
    px = x - w2 
    y_min = max(0, py)
    y_max = min(dh, y + h2)
    x_min = max(0, px)
    x_max = min(dw, x + w2)
    if y_max - y_min <= 0 or x_max - x_min <= 0:
      return

    dst[y_min:y_max, x_min:x_max] += 1


  def generateCountMaps(self, coords):
    '''Generates a count map for the provided list of coordinates.  It can
    count at most 256 object within the receptive field.  Beyond that it
    overflows.
    '''

    s = self.config['receptive_field_size']
    pad = s // 2
    unpadded_size = self.config['tile_size']
    target_size = 1 + unpadded_size + 2 * pad
    countMaps = np.zeros((self.config['cls_nb'], target_size, target_size), dtype=np.int16)

    y_min = 0
    y_max = unpadded_size
    x_min = 0
    x_max = unpadded_size
    for coord in coords:
      if coord[1] >= y_min and coord[1] < y_max and coord[2] >= x_min and coord[2] < x_max:
        self.inc_region(countMaps[coord[0]], coord[1] + pad, coord[2] + pad, s, s)

    return np.moveaxis(countMaps, 0, -1).astype(np.float32)
  

  def preprocessExample(self, image, coords, angle, shear_x, shear_y, scale):
    '''This function is meant to be run as a tf.py_func node on a single
    example.  It returns a randomly perturbed and correctly cropped and padded
    image and generates one or multiple targets.

      image, target = tf.py_func(preprocessExample, [image, coords, class_ids],
                                 [tf.float32, tf.float32])

    Args:
      image: A single training image with value range [0, 1].

    Returns:
      A tuple containing an image and a table of coordinates.
    '''
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

    if self.config['draw_border'] and self.config['contextual_pad'] > 0:
      image = self.draw_border(image, self.config['contextual_pad'], self.config['tile_size'])
      
    # data_preparation.imshow(image, coords=coords, save=True, title='%s_preprocessExampleB' % h)
    # t = np.concatenate(np.moveaxis(target, -1, 0))
    # data_preparation.imshow(t, normalize=True, save=True, title='%s_preprocessExampleC' % h)

    return image.astype(np.float32), target

  
  def draw_border(self, image, p, s):
      image[p - 1:p,             p - 1:p + s + 1, 0  ] += .3
      image[p - 1:p,             p - 1:p + s + 1, 1:3] *= .7
      image[p + s:p + s + 1,     p - 1:p + s + 1, 0  ] += .3
      image[p + s:p + s + 1,     p - 1:p + s + 1, 1:3] *= .7
      image[p:p + s,             p - 1:p,         0  ] += .3
      image[p:p + s,             p - 1:p,         1:3] *= .7
      image[p:p + s,             p + s:p + s + 1, 0  ] += .3
      image[p:p + s,             p + s:p + s + 1, 1:3] *= .7
      image[0:p - 1,             :                   ] *= .7
      image[p + s + 1:2 * p + s, :                   ] *= .7
      image[p - 1:p + s + 1,     0:p - 1             ] *= .7
      image[p - 1:p + s + 1,     p + s + 1:2 * p + s ] *= .7
      image = np.clip(image, 0., 1.)
      return image
    

  # def tower_loss(self, scope):
  #   '''Calculate the total loss on a single tower running the CIFAR model.
  #   Args:
  #     scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
  #   Returns:
  #      Tensor of shape [] containing the total loss for a batch of data
  #   '''

  #   # TODO 
  #   self.build_graph(self.graph)
  #   # and initilaize losses with tf.add_to_collection('losses', weight_decay)

  #   # Assemble all of the losses for the current tower only.
  #   losses = tf.get_collection('loss', scope)

  #   # Calculate the total loss for the current tower.
  #   total_loss = tf.add_n(losses, name='total_loss')

  #   # Attach a scalar summary to all individual losses and the total loss; do the
  #   # same for the averaged version of the losses.
  #   for l in losses + [total_loss]:
  #     # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  #     # session. This helps the clarity of presentation on tensorboard.
  #     loss_name = re.sub('%s_[0-9]*/' % self.config['tower_name'], '', l.op.name)
  #     tf.summary.scalar(loss_name, l)

  #   return total_loss


  # def average_gradients(tower_grads):
  #   '''Calculate the average gradient for each shared variable across all towers.
  #   Note that this function provides a synchronization point across all towers.

  #   This function is adapted from 

  #   Args:
  #     tower_grads: List of lists of (gradient, variable) tuples. The outer list
  #       is over individual gradients. The inner list is over the gradient
  #       calculation for each tower.
  #   Returns:
  #      List of pairs of (gradient, variable) where the gradient has been averaged
  #      across all towers.
  #   '''
  #   average_grads = []
  #   for grad_and_vars in zip(*tower_grads):
  #     # Note that each grad_and_vars looks like the following:
  #     #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
  #     grads = []
  #     for g, _ in grad_and_vars:
  #       # Add 0 dimension to the gradients to represent the tower.
  #       expanded_g = tf.expand_dims(g, 0)

  #       # Append on a 'tower' dimension which we will average over below.
  #       grads.append(expanded_g)

  #     # Average over the 'tower' dimension.
  #     grad = tf.concat(axis=0, values=grads)
  #     grad = tf.reduce_mean(grad, 0)

  #     # Keep in mind that the Variables are redundant because they are shared
  #     # across towers. So .. we will just return the first tower's pointer to
  #     # the Variable.
  #     v = grad_and_vars[0][1]
  #     grad_and_var = (grad, v)
  #     average_grads.append(grad_and_var)
  #   return average_grads

  # '''
  # # Calculate the gradients for each model tower.
  # tower_grads = []
  # # tower_losses = []
  # with tf.variable_scope(tf.get_variable_scope()):
  #   for i in range(self.config['num_gpus']):
  #     with tf.device('/gpu:%d' % i):
  #       with tf.name_scope('%s_%d' % (self.config['tower_name'], i)) as scope:
  #         losses, var_lists, optimizers = tower_loss(scope)

  #         # Reuse variables after the first tower.
  #         tf.get_variable_scope().reuse_variables()

  #         # Retain the summaries from the final tower.
  #         summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
  #         grads = [optimizers for (loss, var_list) in zip(losses, var_lists, optimizers)]
  #         tower_grads.append(grads)

  # grads = None
  # if self.config['num_gpus'] > 1:
  #   # Calculate the mean of each gradient. Note that this is the
  #   # synchronization point across all towers.
  #   grads = average_gradients(tower_grads)
  # else:
  #   grads = tower_grads[0]
  # '''
