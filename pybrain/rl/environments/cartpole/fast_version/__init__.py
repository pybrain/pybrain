try:
    import cartpolewrap
except ImportError:
    raise ImportError('This task needs to be compiled. Please use the script: cartpolecompile.py')

from pybrain.rl.environments.cartpole.fast_version.cartpoleenv import FastCartPoleTask