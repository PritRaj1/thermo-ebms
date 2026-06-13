import grain.python as grain
import tensorflow_datasets as tfds
import numpy as np
from ml_collections import ConfigDict
from typing import Any


def get_dataloader(
	name: str, split: str, batch_size: int, is_training: bool
) -> grain.IterDataset:
	"""Helper to build a single grain dataset."""
	source = tfds.data_source(name, split=split)
	ds = grain.MapDataset.source(source)

	if is_training:
		ds = ds.shuffle(seed=42)

	def preprocess(sample: dict[str, Any]) -> dict[str, Any]:
		image = sample["image"].astype(np.float32)
		image = (image / 127.5) - 1.0
		return {"x": image}

	ds = ds.map(preprocess)
	ds = ds.batch(batch_size=batch_size, drop_remainder=is_training)
	return ds.to_iter_dataset()


def get_loaders(data_config: ConfigDict) -> tuple[grain.IterDataset, grain.IterDataset]:
	name = data_config.name
	batch_size = data_config.batch_size
	train_loader = get_dataloader(name, "train", batch_size, is_training=True)
	test_loader = get_dataloader(name, "test", batch_size, is_training=False)
	return train_loader, test_loader
