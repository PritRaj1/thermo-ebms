import flax.linen as nn
import configparser
from functools import partial

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

hidden_units = int(parser['EBM']['EBM_FEATURE_DIM'])
output_dim = int(parser['EBM']['Z_CHANNELS'])
leak = float(parser['EBM']['EBM_LEAK'])

class EBM(nn.Module):

    @nn.compact
    def __call__(self, z):
        
        f = partial(nn.activation.leaky_relu, negative_slope=leak)
        
        z = nn.Dense(hidden_units)(z)
        z = f(z)
        z = nn.Dense(hidden_units)(z)
        z = f(z)
        z = nn.Dense(output_dim)(z)
        z = f(z)
        
        return z
