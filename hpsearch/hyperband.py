import os, copy, sys
import tensorflow as tf
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

# This program implements the technique detailed in
# https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

class Hyperband():

  def __init__(self, get_random_hps, run_model_and_return_perf, max_iter=81, eta=3):
    '''Initializes a new Hyperband hyper-parameter optimizer.
    
    Arugments:
       max_iter: Maximum iterations/epochs per configuration (default=81).
       eta: Defines downsampling rate (default=3).'''

    self.max_iter = max_iter
    self.eta = eta


  def run(self):
    logeta = lambda x: log(x) / log(self.eta)  # Log to base eta
    s_max = int(logeta(self.max_iter))  # Number of unique executions of Successive Halving (minus one).
    B = (s_max + 1) * self.max_iter     # Total number of iterations (without reuse) per execution of
                                        # Succesive Halving (n,r).

    print('Begin Finite Horizon Hyperband hyperparameter search...')

    try:
      for s in reversed(range(s_max + 1)):
        n = int(ceil(B / self.max_iter / (s + 1) * self.eta ** s))  # Initial number of configurations.
        r = self.max_iter * self.eta ** (-s)                        # Initial number of iterations to run
                                                                    # configurations for.

        # Begin Finite Horizon Successive Halving with (n,r).
        T = [get_random_hps() for i in range(n)] 
        for i in range(s + 1):
          # Run each of the n_i configs for r_i iterations and keep best n_i / self.eta.
          n_i = n * self.eta ** (-i)
          r_i = r * self.eta ** i
          val_losses = [run_model_and_return_perf(num_iters=r_i, hps=t) for t in T]
          T = [T[i] for i in argsort(val_losses)[0:int(n_i / self.eta)]]

    except KeyboardInterrupt:
      print('Keyboard interrupt: Hyperband hyperparameter search stopped.')
      
    # if not os.path.exists(config['result_dir_prefix']):
    #    os.makedirs(config['result_dir_prefix'])
    # with open(config['result_dir_prefix'] + '/hb_results.json', 'w') as f:
    #    json.dump(results, f)
