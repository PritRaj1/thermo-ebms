import flax.linen as nn
import configparser
from functools import partial

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

feature_dim = int(parser["GEN"]["GEN_FEATURE_DIM"])
output_dim = int(parser["GEN"]["CHANNELS"])
image_dim = 64 if parser["PIPELINE"]["DATASET"] == "CelebA" else 32
leak = float(parser["GEN"]["GEN_LEAK"])


class GEN(nn.Module):

    def setup(self):

        self.f = partial(nn.activation.leaky_relu, negative_slope=leak)

        def conditional_64(z):
            z = nn.ConvTranspose(
                feature_dim * 2, kernel_size=(4, 4), strides=(2, 2), padding="SAME"
            )(z)
            z = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)(z)
            z = self.f(z)
            z = nn.ConvTranspose(
                output_dim, kernel_size=(4, 4), strides=(2, 2), padding="SAME"
            )(z)
            z = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)(z)
            return z

        def conditional_32(z):
            z = nn.ConvTranspose(
                output_dim, kernel_size=(4, 4), strides=(2, 2), padding="SAME"
            )(z)
            z = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)(z)
            return z

        if image_dim == 64:
            self.conditional_block = conditional_64

        else:
            self.conditional_block = conditional_32

    @nn.compact
    def __call__(self, z):

        z = nn.ConvTranspose(
            feature_dim * 16, kernel_size=(4, 4), strides=(1, 1), padding="VALID"
        )(z)
        z = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)(z)
        z = self.f(z)

        z = nn.ConvTranspose(
            feature_dim * 8, kernel_size=(4, 4), strides=(2, 2), padding="SAME"
        )(z)
        z = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)(z)
        z = self.f(z)

        z = nn.ConvTranspose(
            feature_dim * 4, kernel_size=(4, 4), strides=(2, 2), padding="SAME"
        )(z)
        z = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)(z)
        z = self.f(z)

        z = self.conditional_block(z)
        z = nn.tanh(z)

        return z
