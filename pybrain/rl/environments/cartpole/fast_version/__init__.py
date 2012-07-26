try:
    import cartpolewrap
except ImportError:
    raise ImportError('This task needs to be compiled. Please use the script: cartpolecompile.py')

from .cartpoleenv import FastCartPoleTask