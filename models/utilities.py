from functools import wraps
import tensorflow as tf
import numpy as np
import operator
import signal
import json
import copy
import math
import sys
import os


class RunningAverage(object):

  def __init__(self, tag, x, summary_fn=tf.summary.scalar, summary_args=(), scope=None):
    """
    Initializes an Average.

    Arguments
      x: Tensor to be averaged over multiple runs.
      tag: Tag for the summary.
      summary_fn: Function used for creating a summary.
      summary_args: Arguments passed to the summary function.
    """
    
    with tf.variable_scope(scope or type(self).__name__):
      counter = tf.Variable(name="counter", initial_value=tf.constant(0),
                            dtype=tf.int32, trainable=False)
      running_sum = tf.Variable(name="running_sum", initial_value=tf.constant(0.),
                                dtype=tf.float32, trainable=False)

      self._running_average = running_sum / tf.cast(counter, tf.float32)
      self._summary = summary_fn(tag or x.name + '_avg', self._running_average, **summary_args)
      self._update_op = tf.group(counter.assign_add(1), running_sum.assign_add(x))
      self._reset_op = tf.group(counter.assign(0), running_sum.assign(0.))
      
      
  @property
  def value(self):
    return self._running_average

  
  @property
  def summary(self):
    return self._summary


  @property
  def update_op(self):
    return self._update_op


  @property
  def reset_op(self):
    return self._reset_op


class NumPyCompatibleJSONEncoder(json.JSONEncoder):
  """See https://stackoverflow.com/a/27050186/852592
  """
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    elif isinstance(obj, np.floating):
      return float(obj)
    else:
      return super(MyEncoder, self).default(obj)


def non_interruptable(f):
  @wraps(f)
  def wrapped(*args, **kwargs):
      s = signal.signal(signal.SIGINT, signal.SIG_IGN)
      r = f(*args, **kwargs)
      signal.signal(signal.SIGINT, s)
      return r
  return wrapped


class Rect(object):
  def __init__(self, min_y, min_x, max_y, max_x, h, w):
    self.min_y = min_y
    self.min_x = min_x
    self.max_y = max_y
    self.max_x = max_x
    self.h = h
    self.w = w

  @property
  def height(self):
    return self.max_y - self.min_y + 1
    
  @property
  def width(self):
    return self.max_x - self.min_x + 1
    
  def __or__(self, other):
    return Rect(min(self.min_y, other.min_y), min(self.min_x, other.min_x),
                max(self.max_y, other.max_y), max(self.max_x, other.max_x),
                self.h, self.w)

  def __repr__(self):
    return '<Rect min_y=%i, min_x=%i, max_y=%i, max_x=%i, h=%i, w=%i>' \
        % (self.min_y, self.min_x, self.max_y, self.max_x, self.h, self.w)


def get_node(name):
  return tf.get_default_graph().as_graph_element(name.split(":")[0])
    
    
def calc_projective_field(input_elem, output_elem, r_in, debug=False, border=False):  
  input_node = get_node(input_elem)
  output_node = get_node(output_elem)
  elem_stack = [output_node.name]
  parents = {output_node.name: None}
  messages = {}
  cache = {}
  
  def init_or_append(d, name, obj):
    if name not in d:
      d[name] = [obj]
    else:
      d[name] += [obj]

  def reduce_union(l):
    l = [x for x in l if x is not None]
    if l:
      r_out = l[0]
      for r in l:
        r_out = r | r_out
      return r_out
    else:
      return None
    
  def send_to_parent(child, message):
    init_or_append(messages, parents[child].pop(), message)
    if child not in cache:
      cache[child] = message

  while elem_stack:
    curr_elem = elem_stack[-1]
    curr_node = get_node(curr_elem)
    
    if debug:
      print('=> Visiting node \'%s\'' % curr_node.name)
      print('Messages', messages)
      print('Parents', parents)
      print('Stack', elem_stack)
    
    if curr_node.name == input_node.name:
      # Base case: Send the queried input coordinates.
      send_to_parent(curr_node.name, r_in)
      elem_stack.pop()
      continue
    
    if curr_node.name in cache:
      # This node has been computed before, so we can reuse its result.
      send_to_parent(curr_node.name, cache[curr_node.name])
      elem_stack.pop()
      continue

    if curr_node.name in messages:
      # By construction of the stack, all child nodes will have been considered
      # before, so once the message key for the current node exists, it contains all
      # messages.
      r = reduce_union(messages[curr_node.name])
      
      if r is not None and \
        (curr_node.type == 'Conv2D' \
        or curr_node.type == 'MaxPool' \
        or curr_node.type == 'AvgPool'):
        s = curr_node.node_def.attr['strides'].list.i[1:3]
        p = curr_node.node_def.attr['padding'].s.decode()
        if curr_node.type == 'Conv2D':
          k = [dim.value for dim in curr_node.inputs[1].shape[:2]]
        else:
          k = curr_node.node_def.attr['ksize'].list.i[1:3]
        
        if debug:
          print('Strides: %s, Padding: %s, Kernel Size: %s' % (s, p, k))
          
        if p == 'VALID':
          out_height = math.ceil(float(r.h - k[0] + 1) / float(s[0]))
          out_width  = math.ceil(float(r.w - k[1] + 1) / float(s[1]))
          pad_top = 0
          pad_left = 0
        elif p == 'SAME':
          out_height = math.ceil(float(r.h) / float(s[0]))
          out_width  = math.ceil(float(r.w) / float(s[1]))
          pad_along_height = max((out_height - 1) * s[0] + k[0] - r.h, 0)
          pad_along_width = max((out_width - 1) * s[1] + k[1] - r.w, 0)
          pad_top = pad_along_height // 2
          pad_bottom = pad_along_height - pad_top
          pad_left = pad_along_width // 2
          pad_right = pad_along_width - pad_left
          
        kr = [float(k[0] - 1) / 2., float(k[1] - 1) / 2.]
        # First, we need to shift the coordinate by the pad. Then we subtract half
        # of the kernel size - 1, to get into the space of the strided output
        # (scaled by stride). Finally, we subtract the same amount to get the longest
        # connection to the left and ceil to obtain the furthest neuron whith such a
        # connection. The same is done in the other direction.
        
        r = Rect(max(math.ceil(float(r.min_y + pad_top - (k[0] - 1)) / float(s[0])), -1 if border else 0),
                 max(math.ceil(float(r.min_x + pad_left - (k[1] - 1)) / float(s[1])), -1 if border else 0),
                 min(math.floor(float(r.max_y + pad_top) / float(s[0])), out_height),
                 min(math.floor(float(r.max_x + pad_left) / float(s[1])), out_width),
                 out_height, out_width)
      elif curr_node.type == 'AtrousConv2D':
        raise ValueError('AtrousConv2D is currently not supported.')
      
      # If it is not a conv layer, simply pass on the message.
      if curr_node.name in parents and parents[curr_node.name] is not None:
        send_to_parent(curr_node.name, r)
      else:
        # At root node.
        return r
      
      # The current node has received all messages and can be popped.
      elem_stack.pop()
      continue
    
    non_control_inputs = [inp for inp in curr_node.inputs]
    control_inputs = [inp for inp in curr_node.control_inputs]
    all_inputs = set(non_control_inputs + control_inputs)
    
    if not all_inputs:
      # Send None at leaves.
      send_to_parent(curr_node.name, None)
      elem_stack.pop()
    else:
      for inp in all_inputs:
        inp_node = get_node(inp.name)
        elem_stack.append(inp_node.name)
        init_or_append(parents, inp_node.name, curr_node.name)

        
def gray2rgb(im):
    w, h = im.shape
    output = np.empty(im.shape[:2] + (3,), dtype=im.dtype)
    output[..., 0] = im
    output[..., 1] = im
    output[..., 2] = im
    return ret
