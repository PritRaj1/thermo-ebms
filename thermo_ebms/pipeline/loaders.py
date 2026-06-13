import grain.python as grain
import tensorflow_datasets as tfds
import numpy as np
from ml_collections import ConfigDict
from typing import Any


def get_dataloader(name: str, split: str, batch_size: int) -> grain.IterDataset:
	"""Helper to build a single grain dataset."""
	is_training = split == "train"
	source = tfds.data_source(name, split=split)
	ds = grain.MapDataset.source(source)

	size = -1
	if is_training:
		ds = ds.shuffle(seed=42)

		bldr = tfds.builder(name)
		size = bldr.info.splits["train"].num_examples

	def preprocess(sample: dict[str, Any]) -> dict[str, Any]:
		image = sample["image"].astype(np.float32)
		image = (image / 127.5) - 1.0
		return {"x": image}, size

	ds = ds.map(preprocess)
	ds = ds.batch(batch_size=batch_size, drop_remainder=is_training)
	return ds.to_iter_dataset(), size


def get_loaders(data_config: ConfigDict) -> tuple[grain.IterDataset, grain.IterDataset]:
	name = data_config.dataset
	batch_size = data_config.batch_size
	train_loader, num_data = get_dataloader(name, "train", batch_size)
	test_loader = get_dataloader(name, "test", batch_size)[0]
	return train_loader, test_loader, num_data // batch_size
