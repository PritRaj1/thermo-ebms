from torchvision import datasets, transforms
import jax
import numpy
import jax.numpy as jnp
from torch.utils import data
from jax.tree_util import tree_map
import numpy as np


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def get_data(name):

    img_dim = 64 if name == "CelebA" else 32

    class Resize_Normalise(object):
        def __call__(self, pic):

            # Resize
            pic = pic.resize((img_dim, img_dim))

            # Cast to [0, 1]
            pic = np.array(pic, dtype=jnp.float32) / 255

            # Normalise to [-1, 1]
            pic = (pic - 0.5) / 0.5

            return pic

    if name == "CIFAR10":
        data = {
            "train": datasets.CIFAR10(
                root="dataset/", train=True, download=True, transform=Resize_Normalise()
            ),
            "test": datasets.CIFAR10(
                root="dataset/",
                train=False,
                download=True,
                transform=Resize_Normalise(),
            ),
        }
    elif name == "SVHN":
        data = {
            "train": datasets.SVHN(
                root="dataset/",
                split="train",
                download=True,
                transform=Resize_Normalise(),
            ),
            "test": datasets.SVHN(
                root="dataset/",
                split="test",
                download=True,
                transform=Resize_Normalise(),
            ),
        }
    elif name == "CelebA":
        data = {
            "train": datasets.CelebA(
                root="dataset/",
                split="train",
                download=True,
                transform=Resize_Normalise(),
            ),
            "test": datasets.CelebA(
                root="dataset/",
                split="test",
                download=True,
                transform=Resize_Normalise(),
            ),
        }
    else:
        raise ValueError("Invalid dataset name.")

    return data["train"], data["test"]


def make_grid(images, n_row=4, padding=1, pad_value=0):
    """Make a grid of images."""

    # Upscale [-1, 1] to [0, 1] to avoid clipping
    images = (images + 1) / 2

    n_images = images.shape[0]
    n_col = int(np.ceil(n_images / n_row))
    grid = np.full(
        (
            images.shape[1] * n_row + padding * (n_row - 1),
            images.shape[2] * n_col + padding * (n_col - 1),
            images.shape[3],
        ),
        pad_value,
        dtype=images.dtype,
    )
    for i in range(n_images):
        row = i // n_col
        col = i % n_col
        grid[
            row * (images.shape[1] + padding) : row * (images.shape[1] + padding)
            + images.shape[1],
            col * (images.shape[2] + padding) : col * (images.shape[2] + padding)
            + images.shape[2],
        ] = images[i]

    return grid
