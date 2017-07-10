import os, copy
import tensorflow as tf
import numpy as np

import models.utilities as utilities

from models import BasicModel

class CountCeptionModel(BasicModel):

  def get_default_config(self):
    config = BasicModel.get_default_config(self)
    model_config = {
      'stride': 1,
      'img_rows_target': 289,
      'img_cols_target': 289,
      'normalize_inputs': True,
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
  

  def build_model_graph(self, inputs, mode):
    image_batch, target_batch, tid_batch = inputs
    
    def selu(x):
      with tf.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0., x, alpha * tf.nn.elu(x))

    def ConvLayer(x, num_filters, kernel_size, name, pad='SAME', is_last=False):
      with tf.variable_scope(name):
        w = tf.get_variable('weights', shape=[kernel_size[0], kernel_size[1],
                                              x.get_shape()[3], num_filters],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=pad)
        if is_last:
          b = tf.get_variable('biases', [num_filters], initializer=tf.zeros_initializer())
          return conv + b
        else:
          # b = tf.get_variable('biases', [num_filters], initializer=tf.zeros_initializer())
          bn = tf.layers.batch_normalization(conv, training=(mode == self.MODE.TRAIN))
          return tf.nn.relu(bn)
          # return selu(conv + b)

    def ConcatBlock(x, num_filters1, num_filters2, name):
      with tf.variable_scope(name):
        conv1x1 = ConvLayer(x, num_filters1, [1, 1], 'conv1x1', pad='VALID')
        conv3x3 = ConvLayer(x, num_filters2, [3, 3], 'conv3x3', pad='SAME')
        return tf.concat([conv1x1, conv3x3], axis=-1)
    
    with tf.name_scope('model'):
      pad = self.config['receptive_field_size'] 
      net = tf.pad(image_batch, [[0, 0], [pad, pad], [pad, pad], [0, 0]] , 'CONSTANT')  
      net = ConvLayer(net, 64, [3, 3], name='conv1', pad='VALID')
      net = ConcatBlock(net, 16, 16, name='concat_block1')
      net = ConcatBlock(net, 16, 32, name='concat_block2')
      net = ConvLayer(net, 16, [14, 14], name='conv2', pad='VALID')
      net = ConcatBlock(net, 112, 48, name='concat_block3')
      net = ConcatBlock(net, 64, 32, name='concat_block4')
      net = ConcatBlock(net, 40, 40, name='concat_block5')
      net = ConcatBlock(net, 32, 96, name='concat_block6')
      net = ConvLayer(net, 32, [17, 17], name='conv3', pad='VALID')
      net = ConvLayer(net, 64, [1, 1], name='conv4', pad='VALID')
      net = ConvLayer(net, 64, [1, 1], name='conv5', pad='VALID')
      net = ConvLayer(net, self.config['cls_nb'], [1, 1], name='out', pad='VALID', is_last=True)
      preds = net

    if mode == self.MODE.TRAIN:
      with tf.name_scope('training'):
        l1_loss = tf.reduce_mean(tf.abs(target_batch - preds))
        tf.add_to_collection(tf.GraphKeys.LOSSES, l1_loss)
        tf.summary.scalar('l1_loss', l1_loss, collections=['train_summaries'])

    if mode == self.MODE.VALIDATE:
      with tf.name_scope('stats'):
        # Mean absolute error
        rf_area = (self.config['receptive_field_size'] / self.config['stride']) ** 2
        pred_counts = tf.reduce_sum(preds / rf_area, axis=[1, 2])
        target_counts = tf.reduce_sum(target_batch / rf_area, axis=[1, 2])
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
      rf_area = (self.config['receptive_field_size'] / self.config['stride']) ** 2
      pred_counts = tf.reduce_sum(preds / rf_area, axis=[1, 2])
      
      self.test_op = (pred_counts, tid_batch)
      
