# The structure of this project is inspired by
# https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3

import os, json, sys, random
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time

if sys.version_info < (3, 0):
    print('This script requires Python 3.')
    sys.exit(1)

# See the __init__ script in the models folder
# `make_models` is a helper function to load any models you have.
from models import make_model 

import os, sys
import time

from hpsearch import hyperband, randomsearch

# Make paths absolute and independent from where the python script is called.
script_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = os.path.join(script_dir, 'results')
sys.path.append(os.path.join(script_dir, 'sealionengine'))

flags = tf.app.flags

# Agent configuration
flags.DEFINE_string('model_name', 'InceptionModel', \
                    'Unique name of the model')
flags.DEFINE_string('config', '{}', \
                    'JSON inputs to fix model parameters, ' \
                    + 'ex: \'{"lr": 0.001}\'')
flags.DEFINE_boolean('load_best_config', False, \
                     'Force to use the best known configuration')

# Run configuration
flags.DEFINE_string('results_dir', '', \
                    'Directory to store/log the model ' \
                    '(if it exist, the model will be loaded from it)')
flags.DEFINE_string('task', 'train', 'The task to run: train, test or search')
flags.DEFINE_boolean('debug', False, 'Debug mode')

def main(_):
  config = flags.FLAGS.__flags.copy()
  config.update(json.loads(config['config']))
  del config['config']
  if config['results_dir'] == '':
    del config['results_dir']

  if config['task'] == 'search':
    # Hyperparameter search cannot be continued, so a new results dir is created.
    config['results_dir'] = os.path.join(results_dir, 'hs', config['model_name'] \
            + time.strftime('_%Y-%m-%d_%H-%M-%S', time.gmtime()))
    hb = Hyperband(config)
    results = hb.run()
  else:
    model = make_model(config)
    if config['task'] == 'train':
      model.train()
    elif config['task'] == 'test':
      model.test()
    else:
      print('Invalid argument: --task=%s. ' \
            + 'It should be either of {train, test, search}.' % config['task'])

if __name__ == '__main__':
  tf.app.run()
